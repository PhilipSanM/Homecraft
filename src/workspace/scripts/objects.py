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
import cv2
# processed folder
processed_folder = '../YOLOv/processed_room/'

# mask folder
mask_folder = '../YOLOv/mask_room/'

# objects folder
objects_folder = '../YOLOv/objects/'

if os.path.exists(object_folder):
    os.system('rm -r ' + object_folder)


objects = os.listdir(mask_folder)


for object in objects:
    image_folders = os.listdir(mask_folder + object)

    for image_folder in image_folders:
        images = os.listdir(mask_folder + object + '/' + image_folder)

        for image in images:
            
            mask_image_path = mask_folder + object + '/' + image_folder + '/' + image
            og_image_path = processed_folder + image_folder + '/' + image 

            # apply mask and get only object image without background

            mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
            og_image = cv2.imread(og_image_path)

            object_image = cv2.bitwise_and(og_image, og_image, mask=mask_image)