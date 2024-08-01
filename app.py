import gradio as gr
import numpy as np
import random
import spaces
import torch
from diffusers import  FluxPipeline, FluxTransformer2DModel,FlowMatchEulerDiscreteScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer,T5EncoderModel, T5TokenizerFast

dtype = torch.bfloat16
device = "cuda"

bfl_repo = "black-forest-labs/FLUX.1-schnell"
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler", revision="refs/pr/1")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype, revision="refs/pr/1")
tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2", torch_dtype=dtype, revision="refs/pr/1")
vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype, revision="refs/pr/1")
transformer = FluxTransformer2DModel.from_pretrained("diffusers-internal-dev/FLUX.1-dev", torch_dtype=dtype)

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = FluxPipeline(
    scheduler=scheduler,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=text_encoder_2,
    tokenizer_2=tokenizer_2,
    vae=vae,
    transformer=transformer,
).to("cuda")

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048

@spaces.GPU()
def infer(prompt, seed=42, randomize_seed=False, width=1024, height=1024, guidance_scale=5.0, num_inference_steps=28, progress=gr.Progress(track_tqdm=True)):
    pipe.to("cuda")
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(seed)
    image = pipe(
            prompt = prompt, 
            width = width,
            height = height,
            num_inference_steps = num_inference_steps, 
            generator = generator,
            guidance_scale=guidance_scale
    ).images[0] 
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
        gr.Markdown(f"""# FLUX.1 [dev]
12B param rectified flow transformer guidance-distilled from [FLUX.1 [pro]](https://blackforestlabs.ai/), [non-commercial license](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md)
[[blog](https://blackforestlabs.ai/2024/07/31/announcing-black-forest-labs/)] [[model](https://huggingface.co/black-forest-labs/FLUX.1-dev)]
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
                    step=1,
                    value=5.0,
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

demo.launch()