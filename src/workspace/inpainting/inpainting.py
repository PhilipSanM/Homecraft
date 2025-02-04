# Copyright 2025 the Regents of the Superior School of Computer Sciene (ESOM) IPN. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid
import os

# mask folder
mask_folder = '../SD/mask_room/'
images_folders = ['images', 'images_2', 'images_4', 'images_8']

# processing folder
processed_folder = '../SD/processed_room/'

# inpainted folder
inpainted_folder = '../SD/inpainted_room/'

# deleting folder if exists
if os.path.exists(inpainted_folder):
    os.system('rm -r ' + inpainted_folder)


# loading model and pipeline

pipeline = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16
)

pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()


generator = torch.Generator(device="cuda").manual_seed(0)

prompt = ""


# find all folders of objects
# objects = os.listdir(mask_folder)
objects = ['chair']

print('detected objects: ', objects)


for object in objects:
    print('Processing object: ', object)
    for folder in images_folders:
        images = os.listdir(mask_folder + object + '/' + folder)

        for image in images:

            og_image = processed_folder + folder + '/' + image
            mask_image = mask_folder + object + '/' + folder + '/' + image

            # print('Processing image: ', og_image)
            # making inpainting

            inpainted_image = pipeline(prompt=prompt,
                                        image=load_image(og_image),
                                        mask_image=load_image(mask_image),
                                        generator=generator,
                                        guidance_scale=8.0,
                                        num_inference_steps=20,  # steps between 15 and 30 work well for us
                                        strength=0.99,
                                        ).images[0]
            

            img_folder = inpainted_folder + object + '/' + folder
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)
            # saving image
            inpainted_image.save(img_folder + '/' + image)









# pipeline = AutoPipelineForInpainting.from_pretrained(
#     "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16
# )

# pipeline.enable_model_cpu_offload()
# # remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
# pipeline.enable_xformers_memory_efficient_attention()


# init_image = load_image("test3.jpg")
# mask_image = load_image("mask3.png")


# generator = torch.Generator(device="cuda").manual_seed(0)

# prompt = ""
# negative_prompt = "high quality, realistic"
# image = pipeline(prompt=prompt, 
#                  image=init_image, 
#                  mask_image=mask_image, 
#                  generator=generator,
#                  guidance_scale=8.0,
#                  num_inference_steps=20,  # steps between 15 and 30 work well for us
#                  strength=0.99, 
#                  ).images[0]


# image.save("inpainting4.png")