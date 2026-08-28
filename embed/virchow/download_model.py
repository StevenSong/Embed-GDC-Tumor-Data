"""Pull the Virchow weights into the image at build time.

paige-ai/Virchow is gated, so the build needs a Hugging Face token with access to
the repo. It is passed as a BuildKit secret (never an ARG, which would end up in
the image history) -- see the Dockerfile and README.

The weights land in $HF_HOME, which the runtime reads with HF_HUB_OFFLINE=1, so
the container never talks to huggingface.co after the build.
"""

import torch

from virchow import EMBED_DIM, TILE_PX, WrappedVirchow


def main():
    model = WrappedVirchow()
    model.eval()

    # smoke test on CPU: the build host has no GPU, but this proves the weights
    # are complete and the graph runs before we ship the image
    x = torch.randint(0, 256, (1, 3, TILE_PX, TILE_PX), dtype=torch.uint8)
    with torch.inference_mode():
        out = model(x)

    assert out.shape == (1, EMBED_DIM), f"unexpected output shape {tuple(out.shape)}"
    print(f"virchow ok: {tuple(out.shape)} {out.dtype}")


if __name__ == "__main__":
    main()
