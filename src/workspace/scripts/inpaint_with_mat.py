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

import time

start = time.time()

# objects folder
OBJECTS_FOLDER = "../MAT/objects/background/images"

def upscale_images_in_folder(folder, output_folder, size=(1080, 1920)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for image_name in os.listdir(folder):
        image_path = os.path.join(folder, image_name)
        image = Image.open(image_path)
        image = image.resize(size)
        output_path = os.path.join(output_folder, image_name)
        image.save(output_path)

# Definir el comando como una lista
command = [
    "python", "../MAT/MAT/generate_image.py",
    "--network", "../MAT/MAT/pretrained/Places_512_FullData.pkl",
    "--dpath", "../MAT/processed_room/images_copy",
    "--mpath", "../MAT/mask_room/masks/images",
    "--outdir", OBJECTS_FOLDER
]


# 
subprocess.run(command, check=True)


upscale_images_in_folder(OBJECTS_FOLDER, OBJECTS_FOLDER)
# python generate_image.py --network pretrained/Places_512_FullData.pkl --dpath test_sets/test/images --mpath test_sets/test/masks --outdir objects/background/images

end = time.time()

print("Inpainting finished in: ", end - start)


# docker exec -it MAT_container bash -c "python ../MAT/scripts/inpaint_with_mat.py"

# docker-compose -f "./src/inpainting_mat.yaml" down