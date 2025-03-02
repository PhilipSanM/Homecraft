import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

# pipeline = AutoPipelineForInpainting.from_pretrained(
#     "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
# )

pipeline = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16
)

pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()


init_image = load_image("test3.jpg")
mask_image = load_image("mask3.png")


generator = torch.Generator(device="cuda").manual_seed(0)

prompt = ""
# negative_prompt = "high quality, realistic"
image = pipeline(prompt=prompt, 
                 image=init_image, 
                 mask_image=mask_image, 
                 generator=generator,
                 guidance_scale=8.0,
                 num_inference_steps=20,  # steps between 15 and 30 work well for us
                 strength=0.99, 
                 ).images[0]


image.save("inpainting4.png")