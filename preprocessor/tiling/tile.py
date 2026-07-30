import argparse
import os

import slideflow as sf
from slideflow.slide import qc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-svs", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main(
    *,  # enforce kwargs
    input_svs: str,
    output_dir: str,
):
    wsi = sf.WSI(
        input_svs,
        tile_px=224,  # input size for Virchow, tile directly to size rather than resize in transform
        tile_um=112,  # assuming 20x is 0.5um per pixel, then whole tile is 112 um
        stride_div=1,  # no overlap
        enable_downsample=True,
        roi_method="ignore",  # no ROIs; extract across the whole slide
    )

    # --- Otsu QC (slide-level background masking) ---
    qc_mask = wsi.qc(qc.Otsu())
    qc_mask.save(
        os.path.join(output_dir, "qc_mask.png")
    )  # eyeball this before committing

    # Optional: preview the tile grid that survives QC, without writing anything
    wsi.preview().save(os.path.join(output_dir, "preview.png"))
    print(f"{len(wsi.coord)} candidate tiles")  # sanity check

    # --- Extract ---
    wsi.extract_tiles(
        tiles_dir=output_dir,  # loose JPGs, one subdir per slide
        img_format="jpg",
        grayspace_fraction=1,  # disable tile-level grayspace filter (Otsu covers it)
        whitespace_fraction=1,  # disable whitespace filter
        report=True,
    )


if __name__ == "__main__":
    args = parse_args()
    main(input_svs=args.input_svs, output_dir=args.output_dir)
