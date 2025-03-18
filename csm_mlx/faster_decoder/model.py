from typing import final

import mlx.core as mx
import mlx.nn as nn
from typing_extensions import override


@final
class FasterAttnDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @override
    def __call__(
        self,
        x: mx.array,    # (b, c:16, d:2048)
    ) -> mx.array:      # (b, )
        return x
