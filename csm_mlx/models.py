from dataclasses import dataclass

import mlx.core as mx
from mlx import nn
from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.llama import LlamaModel

from csm_mlx.config import BACKBONE_CONFIGURATION, DECODER_CONFIGURATION


@dataclass
class ModelArgs(BaseModelArgs):
    backbone_name: str
    decoder_name: str
    n_text_vocab: int
    n_audio_vocab: int
    n_audio_codebooks: int


def csm_1b() -> ModelArgs:
    return ModelArgs(
        backbone_name="1b",
        decoder_name="100m",
        n_text_vocab=128256,
        n_audio_vocab=2051,
        n_audio_codebooks=32,
    )


class CSM(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        # Configuration
        self.n_text_vocab = args.n_text_vocab
        self.n_audio_vocab = args.n_audio_vocab
        self.n_audio_codebooks = args.n_audio_codebooks

        self.n_backbone_embedding = BACKBONE_CONFIGURATION[
            args.backbone_name
        ].num_attention_heads * (
            BACKBONE_CONFIGURATION[args.backbone_name].head_dim or 0
        )
        self.n_decoder_embedding = DECODER_CONFIGURATION[
            args.decoder_name
        ].num_attention_heads * (DECODER_CONFIGURATION[args.decoder_name].head_dim or 0)

        # Model construction
        self.backbone = LlamaModel(BACKBONE_CONFIGURATION[args.backbone_name])
        self.decoder = LlamaModel(DECODER_CONFIGURATION[args.decoder_name])

        self.text_embeddings = nn.Embedding(
            args.n_text_vocab, self.n_backbone_embedding
        )
        self.audio_embeddings = nn.Embedding(
            args.n_audio_vocab * args.n_audio_codebooks, self.n_backbone_embedding
        )
        self.projection = nn.Linear(
            self.n_backbone_embedding, self.n_decoder_embedding, bias=False
        )
        self.codebook0_head = nn.Linear(
            self.n_backbone_embedding, args.n_audio_vocab, bias=False
        )
        self.audio_head = mx.zeros(
            (args.n_audio_codebooks - 1, self.n_decoder_embedding, args.n_audio_vocab)
        )

        # Patch embeddings
        self.backbone.embed_tokens = nn.Identity()  # type: ignore
        self.decoder.embed_tokens = nn.Identity()  # type: ignore

    def embed_audio(self, codebook: int, tokens: mx.array) -> mx.array:
        return self.audio_embeddings(tokens + codebook * self.n_audio_vocab)

    def embed_tokens(self, tokens: mx.array) -> mx.array:
        text_embeds = mx.expand_dims(self.text_embeddings(tokens[:, :, -1]), axis=-2)

        audio_tokens = tokens[:, :, :-1] + (
            self.n_audio_vocab * mx.arange(self.n_audio_codebooks)
        )
        audio_embeds = self.audio_embeddings(audio_tokens.flatten()).reshape(
            (*tokens.shape[:2], self.n_audio_codebooks, -1)
        )

        return mx.concat([audio_embeds, text_embeds], axis=-2)
