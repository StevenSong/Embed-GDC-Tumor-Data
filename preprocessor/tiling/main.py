import argparse
import asyncio
import logging
import multiprocessing as mp
from pathlib import Path

import h5py
import numpy as np
import slideflow as sf
import slideflow.slide.qc as sf_qc
import slideflow.util as sf_util
from kserve import InferInput, InferRequest, Model, ModelServer, model_server
from kserve.inference_client import InferenceGRPCClient

logger = logging.getLogger(__name__)

# Virchow Tiling
TILE_PX = 224
TILE_UM = 112  # Virchow works at 20x i.e. 0.5 MPP
EMBED_DIM = 2560
# must match Triton config.pbtxt; libtorch requires the INPUT__<idx> convention
INPUT_NAME = "INPUT__0"
OUTPUT_NAME = "OUTPUT__0"
GRPC_MAX_MSG = 128 * 1024 * 1024

_CTX = mp.get_context("forkserver")
POOL = None  # module-level; assigned in __main__


class WSITransformer(Model):
    def __init__(
        self,
        name,
        predictor_host,
        tile_batch,
        concurrent_calls_to_predictor,
        queue_depth,
    ):
        super().__init__(name)
        self.predictor_host = predictor_host
        self.tile_batch = tile_batch
        self.concurrent_calls_to_predictor = concurrent_calls_to_predictor
        self.queue_depth = queue_depth
        self._client = None
        self.ready = True

    def _grpc(self):
        # Lazy: aio channels must be built inside the running loop, not at import.
        if self._client is None:
            self._client = InferenceGRPCClient(
                url=self.predictor_host,
                channel_args=[
                    ("grpc.max_send_message_length", GRPC_MAX_MSG),
                    ("grpc.max_receive_message_length", GRPC_MAX_MSG),
                ],
            )
        return self._client

    # ---------- 1. resolve input ------------------------------------------
    async def preprocess(self, payload, headers=None):
        assert isinstance(payload, dict)

        # TODO: replace with Kafka/CloudEvent path
        # KServe unwraps the event and so handle S3 notification body instead
        # For demo, assume input comes preloaded in container
        slide_path = payload["slide_uri"]
        print(f"HERE: {slide_path}")
        return {"slide_path": slide_path, "slide_uri": payload["slide_uri"]}

    # ---------- 2. tile + fan out ------------------------------------------
    async def predict(self, payload, headers=None, response_headers=None):
        assert isinstance(payload, dict)
        loop = asyncio.get_running_loop()

        wsi = sf.WSI(
            payload["slide_path"],
            tile_px=TILE_PX,
            tile_um=TILE_UM,
            stride_div=1,  # no overlap
            enable_downsample=True,
            use_edge_tiles=False,
            roi_method="ignore",  # no ROIs; extract across the whole slide
        )
        wsi.qc(sf_qc.Otsu())

        # preallocate tile embedding buffer
        # actual extracted tiles may be less than n_max, depending on config
        # so keep track of how many tiles were actually written
        n_max = len(wsi.get_tile_dataframe())
        out = np.empty((n_max, EMBED_DIM), dtype=np.float32)
        coords = np.empty((n_max, 2), dtype=np.int32)
        written = 0

        q = asyncio.Queue(maxsize=self.queue_depth)
        client = self._grpc()

        def produce():
            gen = wsi.build_generator(
                shuffle=False,
                deterministic=True,
                whitespace_fraction=1,  # disable, just use Otsu grid
                grayspace_fraction=1,  # disable, just use Otsu grid
                img_format="numpy",
                lazy_iter=True,  # keep memory usage down
                pool=POOL,  # use prebuilt pool
                show_progress=True,  # TODO: disable for prod
            )
            batch_tiles, batch_coords, start = [], [], 0
            assert gen is not None
            for tile in gen():
                tile_im = tile["image"].transpose(2, 0, 1)  # (C, H, W)
                batch_tiles.append(tile_im)
                batch_coords.append(tile["grid"])
                if len(batch_tiles) == self.tile_batch:
                    asyncio.run_coroutine_threadsafe(
                        q.put((start, np.stack(batch_tiles), np.array(batch_coords))),
                        loop,
                    ).result()
                    start += len(batch_tiles)
                    batch_tiles, batch_coords = [], []
            if batch_tiles:  # stragglers
                asyncio.run_coroutine_threadsafe(
                    q.put((start, np.stack(batch_tiles), np.array(batch_coords))),
                    loop,
                ).result()
                start += len(batch_tiles)
            nonlocal written
            written = start

            # signal end of work queue
            for _ in range(self.concurrent_calls_to_predictor):
                asyncio.run_coroutine_threadsafe(q.put(None), loop).result()

        async def consume():
            while (item := await q.get()) is not None:
                (
                    start,  # start idx
                    batch_tiles,  # (B, C, H, W)
                    batch_coords,  # (B, 2)
                ) = item
                inp = InferInput(
                    name=INPUT_NAME,
                    shape=list(batch_tiles.shape),
                    datatype="UINT8",  # serialized model should handle cast/norm
                )
                inp.set_data_from_numpy(batch_tiles, binary_data=True)
                resp = await client.infer(
                    InferRequest(model_name=self.name, infer_inputs=[inp])
                )
                emb = resp.outputs[0].as_numpy()
                out[start : start + len(emb)] = emb
                coords[start : start + len(batch_coords)] = batch_coords

        await asyncio.gather(
            loop.run_in_executor(None, produce),
            *[consume() for _ in range(self.concurrent_calls_to_predictor)],
        )

        # only return the tiles that were actually computed
        return {
            "features": out[:written],
            "coords": coords[:written],
            "meta": payload,
        }

    # ---------- 3. serialize ------------------------------------------------
    async def postprocess(self, result, headers=None, response_headers=None):
        assert isinstance(result, dict)
        meta = result["meta"]
        out_dir = Path("/out")
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"{Path(meta['slide_path']).stem}.h5"

        await asyncio.to_thread(self._write_h5, out_path, result, meta)
        # TODO: upload out_path to s3 and return that URI instead of a local path.

        return {
            "slide_uri": meta["slide_uri"],
            "features_uri": str(out_path),
            "n_tiles": len(result["coords"]),
            "embed_dim": EMBED_DIM,
            "dtype": "float32",
            "tile_px": TILE_PX,
            "tile_um": TILE_UM,
            "encoder": self.name,
        }

    @staticmethod
    def _write_h5(path, result, meta):
        with h5py.File(path, "w") as f:
            f.create_dataset("features", data=result["features"], compression="gzip")
            f.create_dataset("coords", data=result["coords"])


parser = argparse.ArgumentParser(parents=[model_server.parser])
parser.add_argument("--tile_batch", type=int, default=32)
parser.add_argument("--concurrent_calls_to_predictor", type=int, default=8)
parser.add_argument("--num_tile_readers", type=int, default=8)
parser.add_argument("--queue_depth", type=int, default=16)
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    # mirror slideflow internal pool construction but do so globally
    # and before any asyncio threads spin up
    POOL = _CTX.Pool(
        processes=args.num_tile_readers,
        initializer=sf_util.set_ignore_sigint,
    )
    ModelServer().start(
        [
            WSITransformer(
                args.model_name,
                predictor_host=args.predictor_host,
                tile_batch=args.tile_batch,
                concurrent_calls_to_predictor=args.concurrent_calls_to_predictor,
                queue_depth=args.queue_depth,
            )
        ]
    )
