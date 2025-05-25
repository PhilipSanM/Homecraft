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


import os
import subprocess
import PIL.Image as Image
from distutils.dir_util import copy_tree
import time

start = time.time()

# # objects folder
OBJECTS_FOLDER = "../MAT/objects/background/images"
MASKS_FOLDER = "../MAT/mask_room/masks/images"
PROCESSED_FOLDER = "../MAT/processed_room/"
IMAGES_COPY = "../MAT/processed_room/images_copy"

# objects folder local
# OBJECTS_FOLDER = "../objects/background/"
# MASKS_FOLDER = "../mask_room/masks/images"
# PROCESSED_FOLDER = "../processed_room/"
# IMAGES_COPY = "../processed_room/images_copy"


def upscale_images_in_folder(folder, output_folder, size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for image_name in os.listdir(folder):
        image_path = os.path.join(folder, image_name)
        image = Image.open(image_path)
        image = image.resize(size)
        output_path = os.path.join(output_folder, image_name)
        image.save(output_path)
#upscale_images_in_folder(OBJECTS_FOLDER, OBJECTS_FOLDER, (1920, 1080))
def main():     
    # Obtener los nombres de archivos en processed_room (sin extensión)
    processed_files = {f for f in os.listdir(MASKS_FOLDER) if os.path.isfile(os.path.join(MASKS_FOLDER, f))}
    # Recorrer las imágenes procesadas
    processed_images = PROCESSED_FOLDER + 'images/'
    if not os.path.exists(IMAGES_COPY):
            os.makedirs(IMAGES_COPY)
    copy_tree(processed_images, IMAGES_COPY)
    for filename in os.listdir(IMAGES_COPY):        
        #crear una copia de la imagen 
        
    # elmiminar archivos que no se encuentren en processed_room
        if filename not in processed_files:
            os.remove(os.path.join(IMAGES_COPY, filename))   
    upscale_images_in_folder(IMAGES_COPY, IMAGES_COPY, size=(512, 512))
    # Definir el comando como una lista
     command = [
         "python", "../MAT/MAT/generate_image.py",
         "--network", "../MAT/MAT/pretrained/Places_512_FullData.pkl",
         "--dpath", "../MAT/processed_room/images_copy",
         "--mpath", "../MAT/mask_room/masks/images",
         "--outdir", OBJECTS_FOLDER
     ]

    #command = [
    #    "python", "../MAT/generate_image.py",
    #    "--network", "../MAT/pretrained/Places_512_FullData.pkl",
    #   "--dpath", "../processed_room/images_copy",
    #    "--mpath", "../mask_room/masks/images",
    #    "--outdir", OBJECTS_FOLDER
    #]

    # 
    subprocess.run(command, check=True)


    upscale_images_in_folder(OBJECTS_FOLDER, OBJECTS_FOLDER, (1080, 1920))
    # python generate_image.py --network pretrained/Places_512_FullData.pkl --dpath test_sets/test/images --mpath test_sets/test/masks --outdir objects/background/images

    end = time.time()

    print("Inpainting finished in: ", end - start)


    # docker exec -it MAT_container bash -c "python ../MAT/scripts/inpaint_with_mat.py"

    # docker-compose -f "./src/inpainting_mat.yaml" down
if __name__ == "__main__":
    main()
    
