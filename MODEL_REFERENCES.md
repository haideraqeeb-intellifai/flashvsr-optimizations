# Model references

The FlashVSR v1.1 Tiny Long weights are retrieved from [JunhaoZhuang/FlashVSR-v1.1](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1) at revision `27561b186ded3402d7c975f4fd722e2885b6135f`. The model card identifies the license as Apache-2.0. Access to Hugging Face and acceptance of any current repository terms may be required. `python fetch_assets.py` downloads and verifies all files into `models/FlashVSR-v1.1/`.

| File | Purpose | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `diffusion_pytorch_model_streaming_dmd.safetensors` | Tiny Long denoiser | 5,676,070,392 | `bd28180edcf3446c028e32fc6b731a80bf7e4da2ab4caac3186b9499964d37be` |
| `LQ_proj_in.ckpt` | Low-resolution conditioning | 575,694,948 | `d6d011cdaaba6a52645086caa08fa04124e746f6ca568140a24007591142bfd2` |
| `TCDecoder.ckpt` | Video decoder | 189,018,333 | `e224bdcf2f52745cbf4d393ff5374c2ba09e90285d5d19062d2bf63b915b6161` |

The authoritative download URL for each file is `https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1/resolve/27561b186ded3402d7c975f4fd722e2885b6135f/<file>`. No model weight or adapter parameter file is committed to Git.
