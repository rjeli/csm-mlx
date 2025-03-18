from mlx_lm.models.llama import ModelArgs

BACKBONE_CONFIGURATION = {
    "1b": ModelArgs(
        model_type="llama",
        vocab_size=128_256,
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=64,
        intermediate_size=8192,
        hidden_size=2048,
        rms_norm_eps=1e-5,
        rope_scaling={
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        },
        rope_theta=500_000.0,
        rope_traditional=True,
    )
}

DECODER_CONFIGURATION = {
    "100m": ModelArgs(
        model_type="llama",
        vocab_size=128_256,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=128,
        intermediate_size=8192,
        hidden_size=1024,
        rms_norm_eps=1e-5,
        rope_scaling={
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        },
        rope_theta=500_000.0,
        rope_traditional=True,
    )
}

TOKENIZERS = {
    "audio": {
        "repo_id": "kyutai/moshiko-pytorch-bf16",
        "filename": "tokenizer-e351c8d8-checkpoint125.safetensors",
    },
    "text": {"repo_id": "unsloth/Llama-3.2-1B"},
}
