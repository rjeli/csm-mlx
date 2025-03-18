import sys
from typing import cast, final

import audresample
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import soundfile as sf
from datasets import DownloadMode, IterableDatasetDict, load_dataset
from einops.array_api import rearrange
from huggingface_hub import hf_hub_download
from mlx.utils import tree_reduce
from mlx_lm.sample_utils import make_sampler
from moshi_mlx.models.mimi import Mimi, mimi_202407
from tqdm import tqdm
from typing_extensions import override

from csm_mlx import CSM, csm_1b, generate
from csm_mlx.faster_decoder import get_mimi
from csm_mlx.tokenizers import tokenize_audio


@final
class Mlp(nn.Module):
    def __init__(self, d: int, hd: int, bias: bool = True):
        super().__init__()
        self.norm = nn.RMSNorm(d)
        self.gate = nn.Linear(d, hd, bias=bias)
        self.up = nn.Linear(d, hd, bias=bias)
        self.down = nn.Linear(hd, d, bias=bias)

    @override
    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm(x)
        return self.down(nn.silu(self.gate(x)) * self.up(x))


@final
class Layer(nn.Module):
    def __init__(self, d: int, hd: int, bias: bool = True):
        super().__init__()
        self.norm = nn.RMSNorm(d)
        self.down = nn.Linear(d, hd, bias=bias)
        self.up = nn.Linear(hd, d, bias=bias)

    @override
    def __call__(self, x: mx.array) -> mx.array:
        h = self.norm(x)
        h = self.down(h)
        h = nn.silu(h)
        h = self.up(h)
        return x + h


@final
class FasterDecoder(nn.Module):
    def __init__(self, d: int = 2048, hd: int = 32, nc: int = 16):
        super().__init__()
        # self.norm0 = nn.RMSNorm(d)
        # self.down = nn.Linear(d, hd)
        # self.norm1 = nn.RMSNorm(hd)
        # self.fc = nn.Linear(nc * hd, d)
        # self.mlp = [Mlp(d, int(0.5 * d), bias=True) for _ in range(2)]
        self.layers = [Layer(d, hd) for _ in range(16)]

    @override
    def __call__(self, x: mx.array) -> mx.array:
        # x = nn.silu(self.down(self.norm0(x)))
        # x = rearrange(x, "b c d -> b (c d)")
        # x = nn.silu(self.fc(self.norm1(x)))
        for l in self.layers:
            x = l(x)
        return x


if __name__ == "__main__":
    if True:
        ds = cast(
            IterableDatasetDict,
            load_dataset(path="fixie-ai/common_voice_17_0", name="en", streaming=True),
        )
    else:
        audio, sr = sf.read("../csm_mlx/tests/sky.wav", always_2d=True)
        ds = {"train": [{"audio": {"array": audio.mean(1), "sampling_rate": sr}}]}

    """
    sample = next(iter(ds_train))
    print("sample:", sample)

    print(sample["audio"]["array"].shape[0] / sample["audio"]["sampling_rate"])

    print("exiting")
    sys.exit(0)
    """

    # mimi = get_mimi()

    csm = CSM(csm_1b())
    weight = hf_hub_download(
        repo_id="senstella/csm-1b-mlx", filename="ckpt.safetensors"
    )
    _ = csm.load_weights(weight)

    fd = FasterDecoder()
    fd.set_dtype(mx.bfloat16)
    mx.eval(fd.parameters())
    print("total params:", tree_reduce(lambda acc, x: acc + x.size, fd.parameters(), 0))

    def loss_fn(fd: FasterDecoder, X: mx.array, y: mx.array) -> mx.array:
        return nn.losses.mse_loss(fd(X), y)

    loss_and_grad_fn = nn.value_and_grad(fd, loss_fn)

    opt = optim.AdamW(1e-3)

    avg_loss = 0.0
    save_every = 100

    ds_train = ds["train"].shuffle(seed=0)

    for i, s in enumerate(tqdm(ds_train)):
        audio = s["audio"]
        audio = audresample.resample(
            audio["array"].astype(np.float32), audio["sampling_rate"], 24_000
        )[0]
        # codes = mimi.encode(mx.array(audio)[None, None])[0].T
        audio = mx.array(audio)

        a_tok, a_masks = tokenize_audio(audio)

        a_tok = a_tok[:, :-1] + (2051 * mx.arange(32))
        a_emb = csm.audio_embeddings(a_tok)

        # X = rearrange(a_emb[:, :16], "s c d -> s (c d)")
        X = a_emb[:, :16].sum(-2)
        y = a_emb.sum(-2)

        loss, grads = loss_and_grad_fn(fd, X, y)
        # print(f"{loss=}")
        avg_loss += loss.item()
        opt.update(fd, grads)
        mx.eval(fd.parameters(), opt.state)

        if i % save_every == 0:
            avg_loss /= save_every
            print(f"{avg_loss=}")
            avg_loss = 0
            fd.save_weights(f"fd.{i}.safetensors")

        # print(f"{X.shape=} {y.shape=}")

        # print(f"{a_emb.shape=}") # (s=161, 32, 2048)

        # mag = a_emb.square().sum(-1).sqrt()
        # mag_per_code = mag.mean(0)
        # print(" ".join(f"{x:.2f}" for x in mag_per_code))

        # bb_h = csm.backbone(inp)
        # print(f"{bb_h.shape=}")  # (1, 161, 2048)

        # c0_logits = csm.codebook0_head(bb_h)

    """
    while True:
        X, y = None, None
        loss, grads = loss_and_grad_fn(model, X, y)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)

    audio = generate(
        csm,
        text="Hello from Sesame.",
        speaker=0,
        context=[],
        max_audio_length_ms=10_000,
        sampler=make_sampler(temp=0.8, min_p=0.05),
    )
    """
