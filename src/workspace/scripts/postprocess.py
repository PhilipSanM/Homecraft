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
from PIL import Image
import shutil
from distutils.dir_util import copy_tree



from ultralytics import YOLO
import torch
import cv2
import os

# objects folder
objects_folder = '../YOLOv/objects/'

# OG room foler
room_folder = '../YOLOv/processed_room/'



# width, height
sizes = [(540, 960), (270, 480), (135, 240)]

# (1080, 1920), 
# folder names
folder_names = ['images_2', 'images_4', 'images_8']

# getting the last possible image
processed_folder = '../YOLOv/processed_room/'

images = os.listdir(processed_folder + 'images/')

last_image = images[-1] #frame_00340.png

last_number = int(last_image.split('_')[-1].split('.')[0])


objects = os.listdir(objects_folder)

print('==========================================')
print('working background')
# working with background
background_folder = objects_folder + 'background/'

images = os.listdir(background_folder + 'images/')
model = YOLO('yolov8m-seg')


for image in images:

    results = model(background_folder + 'images/' + image)


    # working with every class obtained
    detected_classes = {}

    for idx, prediction in enumerate(results[0].boxes.xywhn): # change final attribute to desired box format
        cls = int(results[0].boxes.cls[idx].item())
        detected_classes[cls] = results[0].names[cls]


    # Detected classes

    if detected_classes:
        print('Detected classes: ', detected_classes)
        # removing image
        os.remove(background_folder + 'images/' + image)
        print('removing image: ', background_folder + 'images/' + image)
    else:
        print('***********************************No classes detected: ', background_folder + 'images/' + image)


print('==========================================')
print('working with objects')

def create_dummy_image(size, path, number):
    width, height = size
    total_digits = 5
    img = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    
    image_number = '0'*(total_digits - len(str(number))) + str(number)
    
    img.save(path + 'frame_' + image_number + '.png')


# fulling missing images with dummies
for object in objects:
    # print('====================================')
    # print('checking object: ', object)

    object_images = objects_folder + object + '/images/'

    for i in range(1, last_number + 1):
        # if image exist jump

        total_digits = 5
        image_number = '0'*(total_digits - len(str(i))) + str(i)

        if os.path.exists(object_images + 'frame_' + image_number + '.png'):
            # print('-----image already exists: ', i)
            continue
        
        # print('creating dummy image: ', i)
        # create dummy image
        create_dummy_image((1080, 1920), object_images, i)


    # making the other folders with their sizes

    for i in range(len(folder_names)):
        foler = folder_names[i]
        size = sizes[i]

        new_image_folder = objects_folder + object + '/' + foler + '/'
        # creating
        if not os.path.exists(new_image_folder):
            os.makedirs(new_image_folder)

        og_folder = objects_folder + object + '/images/'

        for j in range(1, last_number + 1):
            # copy the OG image and resize it

            total_digits = 5
            image_number = '0'*(total_digits - len(str(j))) + str(j)

            og_image_path = og_folder + 'frame_' + image_number + '.png'
            new_image_path = new_image_folder + 'frame_' + image_number + '.png'

            og_image = Image.open(og_image_path)

            new_image = og_image.resize(size)

            new_image.save(new_image_path)


    # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    # print('copying missing files')
    # copying missing files
    colmap_folder = processed_folder + 'colmap/'

    sparce_path = processed_folder + 'sparse_pc.ply'

    transforms_path = processed_folder + 'transforms.json'

    # Copying all files into new object folder

    copy_tree(colmap_folder, objects_folder + object + '/colmap/')

    shutil.copy(sparce_path, objects_folder + object + '/sparse_pc.ply')
    shutil.copy(transforms_path, objects_folder + object + '/transforms.json')






















# for i in range(300, 341):
#     image_name = prefix + '_00' + str(i) + '.png'
#     img = Image.new('RGBA', (width, height), (255, 255, 255, 0))
#     img.save(folder + image_name)
    
