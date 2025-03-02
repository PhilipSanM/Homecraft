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

# LIBRARIES
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid
import os

import time

# CONSTANTS
# mask folder
MASK_FOLDER = '../SD/mask_room/'


# processing folder
PROCESSED_FOLDER = '../SD/processed_room/'

# inpainted folder
INPAINTED_OBJECTS_FOLDER = '../SD/objects/'


# YOLO YOLO_CLASSES
YOLO_YOLO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus", 6: "train", 7: "truck", 8: "boat", 
    9: "traffic light", 10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird", 15: "cat", 
    16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack", 
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee", 30: "skis", 31: "snowboard", 32: "sports ball", 
    33: "kite", 34: "baseball bat", 35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle", 
    40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich", 
    49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake", 56: "chair", 57: "couch", 
    58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 
    66: "keyboard", 67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book", 
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush"
}

# Yolo classes ordered by size
YOLO_CLASSES_ORDER = [79, 76, 78, 67, 65, 64, 66, 40, 41, 42, 43, 44, 39, 74, 75, 26, 27, 24, 28, 29, 32, 34, 35, 36, 37, 38, 73, 77, 56, 45, 68, 69, 70, 71, 63, 14, 15, 16, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 57, 59, 60, 58, 61, 62, 72, 17, 18, 19, 21, 22, 23, 20, 1, 2, 3, 7, 5, 6, 9, 10, 11, 12, 8, 4]

# size of normal images
WIDTH = 1080
HEIGHT = 1920


# FUNCTIONS
def get_pipeline():
    # MODEL PIPELINE
    pipeline = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16
    )

    # Enable CPU offload if you have a compatible CPU and PyTorch 1.9.0 or higher installed
    # Just if memory is not enough
    pipeline.enable_model_cpu_offload()

    # remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
    pipeline.enable_xformers_memory_efficient_attention()

    # Set the random seed for reproducibility
    generator = torch.Generator(device="cuda").manual_seed(0)

    # Empty prompt for inpainting purposes
    prompt = ""

    return pipeline, generator, prompt


def get_all_objects_in_folder(folder):
    objects = os.listdir(folder)
    # Just chair for now
    objects = ['chair']
    return objects


def main():
    # get pipeline function
    pipeline, generator, prompt = get_pipeline()

    # get all objects in folder
    objects = get_all_objects_in_folder(MASK_FOLDER)

    # print('Objects: ', objects)


    # process all objects
    for object in objects:
        # print('Processing object: ', object)

        images_folders = os.listdir(MASK_FOLDER + object)
        for folder in images_folders:
            images = os.listdir(MASK_FOLDER + object + '/' + folder)

            for image in images:

                og_image = PROCESSED_FOLDER + folder + '/' + image
                mask_image = MASK_FOLDER + object + '/' + folder + '/' + image

                # print('Processing image: ', og_image)
                # making inpainting

                inpainted_image = pipeline(prompt=prompt,
                                            # HEIGHT=512,
                                            # WIDTH=512,
                                            image=load_image(og_image),
                                            mask_image=load_image(mask_image),
                                            generator=generator,
                                            guidance_scale=8.0, #8.5 - 9
                                            num_inference_steps=20,  # steps between 15 and 30 work well for us, numero de inferencias 25
                                            strength=0.99, # menos raras
                                            ).images[0]
                

                img_folder = INPAINTED_OBJECTS_FOLDER + 'background' + '/' + folder
                if not os.path.exists(img_folder):
                    os.makedirs(img_folder)


                # resizing image
                inpainted_image = inpainted_image.resize((WIDTH, HEIGHT))

                # saving image
                inpainted_image.save(img_folder + '/' + image)


if __name__ == '__main__':

    print('Starting inpainting...')

    start = time.time()
    main()
    end = time.time()
    print('Inpainting finished in: ', end - start)



