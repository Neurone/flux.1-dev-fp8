import gradio as gr
import numpy as np
import random
import spaces
from diffusers import AuraFlowPipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = AuraFlowPipeline.from_pretrained(
	"fal/AuraFlow-v0.2",
    torch_dtype=torch.float16
).to("cuda")

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

@spaces.GPU()
def infer(prompt, seed=42, randomize_seed=False, width=1024, height=1024, num_inference_steps=4, progress=gr.Progress(track_tqdm=True)):
    generator = torch.Generator().manual_seed(seed)
    image = pipe(
            prompt = prompt, 
            width = width,
            height = height,
            num_inference_steps = num_inference_steps, 
            generator = generator
    ).images[0] 
    return image, seed
 
examples = [
    "A photo of a lavender cat",
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "An astronaut riding a green horse",
    "A delicious ceviche cheesecake slice",
]

css="""
#col-container {
    margin: 0 auto;
    max-width: 520px;
}
"""

with gr.Blocks(css=css) as demo:
    
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""
        # FLUX.1 Schnell
        Demo of the [FLUX.1 Schnell](https://huggingface.co/fal/AuraFlow) 12B parameters rectified flow transformer distilled from [FLUX.1 Pro](https://blackforestlabs.ai/) for fast generation in 4 steps
        [[blog](https://blackforestlabs.ai/2024/07/31/announcing-black-forest-labs/)] [[model](https://black-forest-labs/FLUX.1-schnell)]]
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
                
  
                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=4,
                )
        
        gr.Examples(
            examples = examples,
            fn = infer,
            inputs = [prompt],
            outputs = [result, seed],
            cache_examples="lazy"
        )

    gr.on(
        triggers=[run_button.click, prompt.submit, negative_prompt.submit],
        fn = infer,
        inputs = [prompt, seed, randomize_seed, width, height, num_inference_steps],
        outputs = [result, seed]
    )

demo.queue().launch()