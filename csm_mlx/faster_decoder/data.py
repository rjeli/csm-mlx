import math
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from tqdm import tqdm

from csm_mlx.faster_decoder import get_mimi

if __name__ == "__main__":
    mimi = get_mimi()
    audio_path = Path(sys.argv[1])

    blk_n = 10
    blk_frames = blk_n * 1920

    def save_codes(i: int, codes):
        codes = mx.concat(codes)
        print(f"writing {codes.shape=} to {i}")
        mx.save_safetensors(
            str(audio_path.with_suffix(f".{i}.safetensors")), {"codes": codes}
        )

    i = 0
    codes: list[mx.array] = []

    with sf.SoundFile(audio_path) as f:
        for blk in tqdm(
            f.blocks(blk_frames, always_2d=True, fill_value=0),
            total=math.ceil(f.frames / blk_frames),
        ):
            blk = mx.array(blk).mean(axis=1)
            code = mimi.encode_step(blk[None, None])
            code = code[0].T
            assert code.shape == (blk_n, 32)
            codes.append(code)

            if blk_n * len(codes) >= 512:
                save_codes(i, codes)
                i += 1
                codes = []
                mx.metal.clear_cache()

    save_codes(i, codes)

    # codes = mx.concat(codes)
    # print(f"{codes.shape=}")
    # print("saving..")
