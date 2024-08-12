# FLUX.1-DEV-FP8 Inference App

Inference app for a FP8-quantized flux1-dev model. **This runs on graphic cards with 16 GB of VRAM**.

This a fork of [FLUX.1-dev's Inference App on Hugging Face](https://huggingface.co/spaces/black-forest-labs/FLUX.1-dev).

## Description

This is the inference app for a FP8 quantized version of flux1-dev that can run on graphic cards with 16 GB of VRAM.

Although it has the Hugging Face's UI, this code is meant to run locally on your machine.

Improvements over the original code:

- Quantization of the model to FP8 at startup (I tried to serialize and reload the model from disk, there's no gain in terms of startup speed)
- Automatically save generated images (a WEBP, without metadata, for sharing and a PNG, with metadata, for archiving)
- Automatically insert metadata into images ([tag list](https://exiv2.org/tags.html))
- Automatically insert inference metadata in JSON format into images (this allows you to recreate the same image later)
- Avoid writing memory dump to disk in case of python crash
- Tracking startup time

**Fun fact**. Using the same parameters for inference, you can check the differences between the images generated by the quantized and the non quantized model. Sometimes they are very marginal, sometimes they are more evident.

For my tests, the non-quantized model is in general **better** and you can see the difference.
This is true when you compare the results directly, but the images are generally so good even with the FP8 model that usually you don't care :D

## Install

```bash
git clone https://github.com/Neurone/flux.1-dev-fp8.git
cd flux.1-dev-fp8
python3 -m virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
cd flux.1-dev-fp8
source .venv/bin/activate
python app.py
```

## Alternative Run

If you experience memory problems from time to time, especially when you try 2048x2048 images, try starting the app with this:

```bash
cd flux.1-dev-fp8
source .venv/bin/activate
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python app.py
```

## Inference Metadata

This is an example of the inference metadata saved into the PNG images.

```json
{
  "model": {
    "name": "flux",
    "id": "flux1-dev-fp8",
    "multihash": "1220dc4a58f44c1ba335822aaf041b2d19483bfd12d5dc260f6fac403f7be5f33181",
    "license_url": "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/LICENSE.md",
    "license_multihash": "1220b7a00498845420da83aad42857f69fbfcf731fd1efa6d1bb596a884f2f2cbf53",
    "author": "Black Forest Labs"
  },
  "input": {
    "prompt": "A majestic angel with large, dark wings, adorned in flowing blue robes, carrying a sleeping baby and surrounded by cherubs in a moonlit sky.  Whimsical, ethereal, celestial, fantasy art",
    "seed": 1262301990,
    "cfg_scale": 3.5,
    "steps": 28,
    "width": 1024,
    "height": 1024,
    "type": "txt2img"
  },
  "output": {
    "filename": "1723504398.6657534.png",
    "format": "image/png",
    "image_multihash": "1220314871dc0bba139548c91895604e16f4183cd2911e825e924244ebee5e0b5916",
    "creation_date_time": "2024-08-12 23:13:18.977696+00:00"
  }
}
```

## Performance

This is my configuration:

- CPU: Intel Core i7-12700K
- GPU: Nvidia GeForce RTX 4080 SUPER 16 GB
- RAM: 64 GB DDR5
- DISK (Operating System): SSD NVMe Crucial P5 Plus 2TB
- DISK (Models): External USB3 HDD
- OPERATING SYSTEM: Ubuntu 22.04.3 LTS
- PYTHON VERSION: 3.10.12

Excluding the real first time when you need to download all the resources, these are some examples of the performance I get.

Prompt: *A majestic angel with large, dark wings, adorned in flowing blue robes, carrying a sleeping baby and surrounded by cherubs in a moonlit sky.  Whimsical, ethereal, celestial, fantasy art*

CFG: 3.5

| operation | time spent |
| - | - |
| Startup time | ~4-5 minutes (the first startup after reboot takes ~13 mins.)|
| Inference; 512x512; 28 steps | 19 seconds |
| Inference; 1024x1024; 28 steps | 45 seconds |
| Inference; 1024x1024; 50 steps | 1 minute and 6 seconds |
| Inference; 1024x2024; 28 steps | 1 minute and 15 seconds |
| Inference; 1024x2048; 40 steps | 1 minute and 44 seconds |
| Inference; 2048x2048; 28 steps | 2 minutes and 39 seconds|
| Inference; 2048x2048; 50 steps | 4 minutes and 46 seconds|

After a couple of runs, most of the startup time is spent in the part **before** the quantization of the model.

Quantization adds ~90 sec at the startup time.

## Samples

### 512x512; 28 steps

![512x512; 28 steps](./samples/1723504145.2547626.webp "512x512; 28 steps")

### 1024x1024; 28 steps

![1024x1024; 28 steps](./samples/1723504398.6657534.webp "1024x1024; 28 steps")

### 1024x2024; 28 steps

![1024x2024; 28 steps](./samples/1723501887.0529025.webp "1024x2024; 28 steps")

### 1024x2048; 40 steps

![1024x2048; 40 steps](./samples/1723504062.1747687.webp "1024x2048; 40 steps")

### 2048x2048; 50 steps

![2048x2048; 50 steps](./samples/1723503836.2297032.webp "2048x2048; 50 steps")

## Example Of Consecutive Startups

```bash
❯ python app.py
./.venv/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:211: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
  @torch.library.impl_abstract("xformers_flash::flash_fwd")
./.venv/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:344: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
  @torch.library.impl_abstract("xformers_flash::flash_bwd")
2024-08-12 23:15:53.585458 Started
Downloading shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 44.69it/s]
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [03:07<00:00, 93.81s/it]
You set `add_prefix_space`. The tokenizer needs to be converted from the slow tokenizers
Fetching 3 files: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 26.87it/s]
2024-08-12 23:19:25.183450 Quantizing transformer
2024-08-12 23:27:55.037217 Quantizing text encoder 2
2024-08-12 23:28:15.598087 Loading demo
Will cache examples in './gradio_cached_examples/19' directory at first use. 


Running on local URL:  http://127.0.0.1:7860
INFO:httpx:HTTP Request: GET http://127.0.0.1:7860/startup-events "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: HEAD http://127.0.0.1:7860/ "HTTP/1.1 200 OK"

To create a public link, set `share=True` in `launch()`.
INFO:httpx:HTTP Request: GET https://checkip.amazonaws.com/ "HTTP/1.1 200 "
INFO:httpx:HTTP Request: GET https://checkip.amazonaws.com/ "HTTP/1.1 200 "
INFO:httpx:HTTP Request: GET https://api.gradio.app/pkg-version "HTTP/1.1 200 OK"
^CKeyboard interruption in main thread... closing server.


### 12:21 mins


❯ python app.py
./.venv/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:211: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
  @torch.library.impl_abstract("xformers_flash::flash_fwd")
./.venv/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:344: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
  @torch.library.impl_abstract("xformers_flash::flash_bwd")
2024-08-12 23:28:37.290645 Started
Downloading shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 44.73it/s]
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [03:07<00:00, 93.78s/it]
You set `add_prefix_space`. The tokenizer needs to be converted from the slow tokenizers
Fetching 3 files: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 28.29it/s]
2024-08-12 23:32:04.014079 Quantizing transformer
2024-08-12 23:34:52.977714 Quantizing text encoder 2
2024-08-12 23:35:14.110980 Loading demo
Will cache examples in './gradio_cached_examples/19' directory at first use. 


Running on local URL:  http://127.0.0.1:7860
INFO:httpx:HTTP Request: GET http://127.0.0.1:7860/startup-events "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: HEAD http://127.0.0.1:7860/ "HTTP/1.1 200 OK"

To create a public link, set `share=True` in `launch()`.
INFO:httpx:HTTP Request: GET https://checkip.amazonaws.com/ "HTTP/1.1 200 "
INFO:httpx:HTTP Request: GET https://checkip.amazonaws.com/ "HTTP/1.1 200 "
INFO:httpx:HTTP Request: GET https://api.gradio.app/pkg-version "HTTP/1.1 200 OK"
^CKeyboard interruption in main thread... closing server.


## 6:37 mins


❯ python app.py
./.venv/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:211: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
  @torch.library.impl_abstract("xformers_flash::flash_fwd")
./.venv/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:344: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
  @torch.library.impl_abstract("xformers_flash::flash_bwd")
2024-08-12 23:36:22.058130 Started
Downloading shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 44.41it/s]
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [02:42<00:00, 81.02s/it]
You set `add_prefix_space`. The tokenizer needs to be converted from the slow tokenizers
Fetching 3 files: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 30.68it/s]
2024-08-12 23:39:19.638047 Quantizing transformer
2024-08-12 23:40:28.817001 Quantizing text encoder 2
2024-08-12 23:40:49.528701 Loading demo
Will cache examples in './gradio_cached_examples/19' directory at first use. 


Running on local URL:  http://127.0.0.1:7860
INFO:httpx:HTTP Request: GET http://127.0.0.1:7860/startup-events "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: HEAD http://127.0.0.1:7860/ "HTTP/1.1 200 OK"

To create a public link, set `share=True` in `launch()`.
INFO:httpx:HTTP Request: GET https://checkip.amazonaws.com/ "HTTP/1.1 200 "
INFO:httpx:HTTP Request: GET https://checkip.amazonaws.com/ "HTTP/1.1 200 "
INFO:httpx:HTTP Request: GET https://api.gradio.app/pkg-version "HTTP/1.1 200 OK"
^CKeyboard interruption in main thread... closing server.


## 4:27 mins


❯ python app.py
/home/developer/workspace/FLUX.1-dev/.venv/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:211: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
  @torch.library.impl_abstract("xformers_flash::flash_fwd")
/home/developer/workspace/FLUX.1-dev/.venv/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:344: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
  @torch.library.impl_abstract("xformers_flash::flash_bwd")
2024-08-12 23:46:18.007443 Started
Downloading shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 73.64it/s]
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [03:09<00:00, 94.50s/it]
You set `add_prefix_space`. The tokenizer needs to be converted from the slow tokenizers
Fetching 3 files: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 26.61it/s]
2024-08-12 23:49:43.073283 Quantizing transformer
2024-08-12 23:51:13.973582 Quantizing text encoder 2
2024-08-12 23:51:34.588930 Loading demo
Will cache examples in '/home/developer/workspace/FLUX.1-dev/gradio_cached_examples/19' directory at first use. 


Running on local URL:  http://127.0.0.1:7860
INFO:httpx:HTTP Request: GET http://127.0.0.1:7860/startup-events "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: HEAD http://127.0.0.1:7860/ "HTTP/1.1 200 OK"

To create a public link, set `share=True` in `launch()`.
INFO:httpx:HTTP Request: GET https://checkip.amazonaws.com/ "HTTP/1.1 200 "
INFO:httpx:HTTP Request: GET https://checkip.amazonaws.com/ "HTTP/1.1 200 "
INFO:httpx:HTTP Request: GET https://api.gradio.app/pkg-version "HTTP/1.1 200 OK"


## 5:16 mins
```

## Utils

Show all exif metadata

```bash
exiftool <filename>
```

Show only inference metadata

```bash
exiftool -usercomment -s3 <filename> | jq
```

Delete metadata from one or all files

```bash
exift -all= <filename>
exift -all= *
```

## Side notes

1. The model_multihash (1220dc4a58f44c1ba335822aaf041b2d19483bfd12d5dc260f6fac403f7be5f33181) is derived from the
serialization of the quantized transformer using the optimum.quanto libray just after the freeze operation.

    ```python
    from optimum.quanto import quantization_map
    from safetensors.torch import save_file
    save_file(transformer.state_dict(), './flux1-dev-transformer-fp8.safetensors')
    ```

2. The `Image.Exif()` object let you set the exif data and save them when saving the image for the first time.
This works correctly unless you need to use the UserComments field, and I want to use it. In that case, there's
an error in the encoding.  To avoid the encoding error, I use an external library (exiv2) that I was able to make it
work only when using an actual file and not when reading the image from memory. This is why there are two saving
steps implemented, and not only once.

## Credits

- Thanks to [Black Forest Labs](https://blackforestlabs.ai/) for releasing the model free to use, and the inference code open source
- Huge thanks to [@AmericanPresidentJimmyCarter](https://gist.github.com/AmericanPresidentJimmyCarter) for developing the [original quantization code](https://gist.github.com/AmericanPresidentJimmyCarter/873985638e1f3541ba8b00137e7dacd9).
