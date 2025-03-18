from pathlib import Path

from huggingface_hub import hf_hub_download
from mlx.utils import tree_reduce
from mlx_lm.utils import make_sampler
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn

from csm_mlx.cli.utils import read_audio, write_audio
from csm_mlx.faster_decoder.train import FasterDecoder
from csm_mlx.generation import generate
from csm_mlx.models import CSM, csm_1b
from csm_mlx.segment import Segment

if __name__ == "__main__":
    csm = CSM(csm_1b())
    weight = hf_hub_download(
        repo_id="senstella/csm-1b-mlx", filename="ckpt.safetensors"
    )
    _ = csm.load_weights(weight)

    fd = FasterDecoder()
    _ = fd.load_weights("models/fd.249700.safetensors")
    print("total params:", tree_reduce(lambda acc, x: acc + x.size, fd.parameters(), 0))
    print(fd)

    sampler = make_sampler(
        temp=0.8,
        top_p=0.0,
        min_p=0.05,
        top_k=-1,
        min_tokens_to_keep=1,
    )

    context = [
        Segment(
            speaker=0,
            text="When I heard the release demo, I was shocked, angered, and in disbelief that Mr. Altman would pursue a voice that sounded so eerily similar to mine that my closest friends and news outlets could not tell the difference.",
            audio=read_audio(Path("../csm_mlx/tests/sky.wav"), 24_000),
        )
    ]

    text = "A mirror and exaltation of the false intellect of the nerd, that never leaves the stream of words, syllogisms, motives and desire, that is always forced and contrived, because its under pressure of some petty need. And its really grotesque."

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total} ({task.speed}/s)"),
    ) as progress:
        ptask = progress.add_task(description="inferring...", total=1)
        result = generate(
            csm,
            text,
            speaker=0,
            context=context,
            max_audio_length_ms=10 * 1000,
            sampler=sampler,
            progress_fn=lambda i, tot: progress.update(ptask, completed=i, total=tot),
            #
            fd=fd,
            actual_codebooks=16,
        )

    write_audio(result, Path("out.wav"), 24_000)
