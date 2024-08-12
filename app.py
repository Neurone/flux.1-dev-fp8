import gradio as gr
import numpy as np
import random
import torch
import datetime
import logging
import atexit
import json
import exiv2
import time

from datetime import timezone
from multiformats import multihash
from optimum.quanto import freeze, qfloat8, quantize
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

logging.basicConfig(level=logging.INFO)

print(datetime.datetime.now(), "Started")

def silence_crash_logging():
    def custom_exit_handler():
        # Handle the exception instead of allowing Python to generate a crash log
        pass
    atexit.register(custom_exit_handler)

silence_crash_logging()

dtype = torch.bfloat16

bfl_repo = "black-forest-labs/FLUX.1-dev"
revision = "refs/pr/3"

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler", revision=revision)
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype, revision=revision)
tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2", torch_dtype=dtype, revision=revision)
vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype, revision=revision)
transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=dtype, revision=revision)

print(datetime.datetime.now(), "Quantizing transformer")
quantize(transformer, weights=qfloat8)
freeze(transformer)

print(datetime.datetime.now(), "Quantizing text encoder 2")
quantize(text_encoder_2, weights=qfloat8)
freeze(text_encoder_2)

pipe = FluxPipeline(
    scheduler=scheduler,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=text_encoder_2,
    tokenizer_2=tokenizer_2,
    vae=vae,
    transformer=transformer,
)
pipe.enable_model_cpu_offload()

print(datetime.datetime.now(), "Loading demo")

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048

def infer(prompt, seed=42, randomize_seed=False, width=1024, height=1024, guidance_scale=5.0, num_inference_steps=28, progress=gr.Progress(track_tqdm=True)):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(seed)
    print(datetime.datetime.now(), f"Image generation started | Prompt: {prompt[:30]} | Seed: {seed} | Res: {width}x{height} | CFG: {guidance_scale} |  Steps: {num_inference_steps}") 
    image = pipe(
        prompt = prompt, 
        width = width,
        height = height,
        num_inference_steps = num_inference_steps, 
        generator = generator,
        guidance_scale=guidance_scale
    ).images[0]
    # Autosave the image
    model_id = "flux1-dev-fp8"
    folder = "./autosave"
    # TODO: create folder if it does not exist
    timestamp = f"{time.time():.7f}"
    # Light format, without metadata, for sharing
    filename = timestamp+".webp"
    filepath = f"{folder}/{filename}"
    image.save(filepath, format="webp")
    print(datetime.datetime.now(), "Image saved to:", filepath) 
    # Best format, with metadata, for archiving
    filename = timestamp+".png"
    filepath = f"{folder}/{filename}"
    image.save(filepath, format="png")
    print(datetime.datetime.now(), "Image saved to:", filepath)
    # Prepare metadata values for the image
    date_time = str(datetime.datetime.now(timezone.utc))
    model_multihash = "1220dc4a58f44c1ba335822aaf041b2d19483bfd12d5dc260f6fac403f7be5f33181"
    model_license_url = "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/LICENSE.md"
    model_license_multihash = "1220b7a00498845420da83aad42857f69fbfcf731fd1efa6d1bb596a884f2f2cbf53"
    model_author = "Black Forest Labs"
    input_type= "txt2img"
    # Compute image sha256
    with open(filepath, 'rb') as file:
        file_data = file.read()
    digest = multihash.digest(file_data, "sha2-256")
    image_multihash = digest.hex()
    # Matedata summary
    json_metadata = json.dumps({
        "model": {
            "name": "flux",
            "id": model_id,
            "multihash": model_multihash,
            "license_url": model_license_url,
            "license_multihash": model_license_multihash,
            "author": model_author
        },
        "input": {
            "prompt": prompt,
            "seed": seed,
            "cfg_scale": guidance_scale,
            "steps": num_inference_steps,
            "width": width,
            "height": height,
            "type": input_type
        },
        "output": {
            "filename": filename,
            "format": "image/png",
            "image_multihash": image_multihash,
            "creation_date_time": date_time
        }})
    # Add metadata
    exiv2_image = exiv2.ImageFactory.open(filepath)
    exiv2_image.readMetadata()
    exif_data = exiv2_image.exifData()
    exif_data["Exif.Image.Make"] = model_author
    exif_data["Exif.Image.Model"] = model_id
    exif_data["Exif.Image.ImageDescription"] = prompt
    exif_data["Exif.Photo.UserComment"] = json_metadata
    exif_data["Exif.Image.DateTime"] = date_time
    exif_data["Exif.Photo.DateTimeOriginal"] = date_time
    exif_data["Exif.Image.Artist"] = model_id
    exif_data["Exif.Image.ImageID"] = image_multihash
    exif_data["Exif.Image.Copyright"] = model_license_url+";"+model_license_multihash
    exiv2_image.writeMetadata()
    print(datetime.datetime.now(), "Metadata updated:", filepath)

    return image, seed
 
examples = [
    "a tiny astronaut hatching from an egg on the moon",
    "a cat holding a sign that says hello world",
    "an anime illustration of a wiener schnitzel",
]

css="""
#col-container {
    margin: 0 auto;
    max-width: 520px;
}
"""

with gr.Blocks(css=css) as demo:
    
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""# FLUX.1 [dev] - Quantized to FP8
This version of flux1-dev is quantized to qfloat8 weights, so you can run the model locally on graphic cards with 16 GB of VRAM.

Original code [here](https://huggingface.co/spaces/black-forest-labs/FLUX.1-dev/tree/main).
        """)
        
        with gr.Row():
            
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            
            run_button = gr.Button("Run", scale=0)
        
        result = gr.Image(label="Result", show_label=False)
        
        with gr.Accordion("Advanced Settings", open=False):
            
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )
            
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            
            with gr.Row():
                
                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                )
                
                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                )
            
            with gr.Row():

                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=1,
                    maximum=15,
                    step=0.1,
                    value=3.5,
                )
  
                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=28,
                )
        
        gr.Examples(
            examples = examples,
            fn = infer,
            inputs = [prompt],
            outputs = [result, seed],
            cache_examples="lazy"
        )

    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn = infer,
        inputs = [prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps],
        outputs = [result, seed]
    )

demo.launch(share=False)
