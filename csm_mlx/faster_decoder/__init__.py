def get_mimi():
    from huggingface_hub import hf_hub_download
    from moshi_mlx.models.mimi import Mimi, mimi_202407

    mimi = Mimi(mimi_202407(32))
    weight = hf_hub_download(
        repo_id="kyutai/moshiko-pytorch-bf16",
        filename="tokenizer-e351c8d8-checkpoint125.safetensors",
    )
    _ = mimi.load_pytorch_weights(weight, strict=False)
    return mimi
