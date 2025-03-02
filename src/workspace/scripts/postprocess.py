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
import os
from PIL import Image
import shutil
from distutils.dir_util import copy_tree
from ultralytics import YOLO
import torch
import cv2

import time

# CONSTANTS
# objects folder
OBJECTS_FOLDER = '../YOLOv/objects/'

# OG room foler
ROOM_FOLDER = '../YOLOv/processed_room/'



# width, height
SIZEZ_FROM_PREPROCESSING = [(540, 960), (270, 480), (135, 240)]

# (1080, 1920), 
# folder names
FOLDER_NAMES_FROM_PREPROCESSING = ['images_2', 'images_4', 'images_8']

# getting the last possible image
PROCESSED_FOLDER = '../YOLOv/processed_room/'

# background folder
BACKGROUND_FOLDER = OBJECTS_FOLDER + 'background/'

def get_last_image_number():
    images = os.listdir(PROCESSED_FOLDER + 'images/')

    last_image = images[-1] #frame_00340.png

    last_number = int(last_image.split('_')[-1].split('.')[0])

    return last_number


def remove_images_with_objects_in_background():

    images = os.listdir(BACKGROUND_FOLDER + 'images/')
    model = YOLO('yolov8m-seg')

    for image in images:

        results = model(BACKGROUND_FOLDER + 'images/' + image)

        # working with every class obtained
        detected_classes = {}

        for idx, prediction in enumerate(results[0].boxes.xywhn): # change final attribute to desired box format
            cls = int(results[0].boxes.cls[idx].item())
            detected_classes[cls] = results[0].names[cls]

        # Detected classes
        if detected_classes:
            # print('Detected classes: ', detected_classes)
            # removing image
            os.remove(BACKGROUND_FOLDER + 'images/' + image)
            # print('removing image: ', BACKGROUND_FOLDER + 'images/' + image)
        else:
            # adding the RGBA channel to the image in background
            img = Image.open(BACKGROUND_FOLDER + 'images/' + image)
            img = img.convert('RGBA')
            img.save(BACKGROUND_FOLDER + 'images/' + image)


def create_dummy_image(size, path, number):
    width, height = size
    total_digits = 5
    img = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    
    image_number = '0'*(total_digits - len(str(number))) + str(number)
    
    img.save(path + 'frame_' + image_number + '.png')

def main():
    
    # working with background
    # print('working background')
    remove_images_with_objects_in_background()


    # working with objects
    # print('working with objects')

    # getting the last possible image
    last_number = get_last_image_number()

    # getting the objects founded
    objects = os.listdir(OBJECTS_FOLDER)
    
    for object in objects:
        
        # Making missing images from 1 to last_number
        OBJECT_IMAGES_FOLDER = OBJECTS_FOLDER + object + '/images/'

        for i in range(1, last_number + 1):
            # if image exist jump

            total_digits = 5
            image_number = '0'*(total_digits - len(str(i))) + str(i)

            if os.path.exists(OBJECT_IMAGES_FOLDER + 'frame_' + image_number + '.png'):
                # print('-----image already exists: ', i)
                continue
            
            # print('creating dummy image: ', i)
            # create dummy image
            create_dummy_image((1080, 1920), OBJECT_IMAGES_FOLDER, i)


        # making the other folders with their SIZEZ_FROM_PREPROCESSING
        for i in range(len(FOLDER_NAMES_FROM_PREPROCESSING)):
            foler = FOLDER_NAMES_FROM_PREPROCESSING[i]
            size = SIZEZ_FROM_PREPROCESSING[i]

            new_image_folder = OBJECTS_FOLDER + object + '/' + foler + '/'
            # creating
            if not os.path.exists(new_image_folder):
                os.makedirs(new_image_folder)

            og_folder = OBJECTS_FOLDER + object + '/images/'

            for j in range(1, last_number + 1):
                # copy the OG image and resize it

                total_digits = 5
                image_number = '0'*(total_digits - len(str(j))) + str(j)

                og_image_path = og_folder + 'frame_' + image_number + '.png'
                new_image_path = new_image_folder + 'frame_' + image_number + '.png'

                og_image = Image.open(og_image_path)

                new_image = og_image.resize(size)

                new_image.save(new_image_path)



        # copying missing files like colmap, sparse_pc.ply, transforms.json
        COLMAP_FOLDER = PROCESSED_FOLDER + 'colmap/'

        SPARCE_FILE_PATH = PROCESSED_FOLDER + 'sparse_pc.ply'

        TRANSFORMS_FILE_PATH = PROCESSED_FOLDER + 'transforms.json'

        # Copying all files into new object folder
        copy_tree(COLMAP_FOLDER, OBJECTS_FOLDER + object + '/colmap/')

        shutil.copy(SPARCE_FILE_PATH, OBJECTS_FOLDER + object + '/sparse_pc.ply')
        shutil.copy(TRANSFORMS_FILE_PATH, OBJECTS_FOLDER + object + '/transforms.json')




if __name__ == '__main__':
    print('Starting postprocessing....')
    start = time.time()
    main()
    end = time.time()
    print('Postprocessing finished in: ', end - start)



