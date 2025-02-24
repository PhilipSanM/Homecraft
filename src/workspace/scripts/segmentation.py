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
from ultralytics import YOLO
import torch
import cv2
import os

import time

# CONSTANTS
YOLO_MODEL = 'yolov8m-seg'


# Workspace from nerfstudio
PROCESSED_FOLDER = '../YOLOv/processed_room/'
IMAGES_FOLDERS = ['images']

# mask folder
MASK_FOLDER = '../YOLOv/mask_room/'

# objects folder
OBJECTS_FOLDER = '../YOLOv/objects/'


# FUNCTIONS
def make_dilatation_2_image(image_path):
    # applying dilatation
    kernel = cv2.MORPH_ELLIPSE
    # kernel = cv2.MORPH_RECT
    # kernel = cv2.MORPH_CROSS
    element = cv2.getStructuringElement(kernel, (2 * 5 + 1, 2 * 5 + 1),
                                        (3, 3))
    dilatation =  cv2.dilate(cv2.imread(image_path), element, iterations=7)
    # 3 iterations 
    cv2.imwrite(image_path, dilatation)



def get_object_from_mask(mask_path, og_image_path, image, masked_file_path):

    # if not folder exists, create it
    if not os.path.exists(masked_file_path):
        os.makedirs(masked_file_path)
    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = mask.astype('uint8')

    # Load original image
    og_image = cv2.imread(og_image_path)

    # Resize mask
    height, width = og_image.shape[:2]
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    # Apply mask
    masked_image = cv2.bitwise_and(og_image, og_image, mask=mask)

    # Remove black background
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2BGRA)
    masked_image[:, :, 3] = mask

    # Save masked image
    masked_file = masked_file_path + '/' + image
    cv2.imwrite(masked_file, masked_image)



def get_mask_from_inference(result, mask_img_path, img_folder, number):
    # get array results
    masks = result.masks.data
    boxes = result.boxes.data

    # extract classes
    clss = boxes[:, 5]
    # get indices of results where class is 0 (people in COCO)
    object_indices = torch.where(clss == number)
    # use these indices to extract the relevant masks
    object_masks = masks[object_indices]

    # scale for visualizing results
    object_mask = torch.any(object_masks, dim=0).int() * 255

    # make folder if not exists
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    
    # save to file
    cv2.imwrite(mask_img_path, object_mask.cpu().numpy())
    

def get_all_classes_obtained(results):
    detected_classes = {}

    for idx, prediction in enumerate(results[0].boxes.xywhn): # change final attribute to desired box format
        cls = int(results[0].boxes.cls[idx].item())
        detected_classes[cls] = results[0].names[cls]

    return detected_classes



def main():

    # deleting folder if exists
    if os.path.exists(OBJECTS_FOLDER):
        os.system('rm -r ' + OBJECTS_FOLDER)


    # deleting folder if exists
    if os.path.exists(MASK_FOLDER):
        os.system('rm -r ' + MASK_FOLDER)

    # load model
    model = YOLO(YOLO_MODEL)


    # For every folder and for every image, generate masks
    for folder in IMAGES_FOLDERS:
        # Path to the folder
        path = PROCESSED_FOLDER + folder
        # print('Processing folder: ', path)

        # Find all images in the folder
        images = os.listdir(path)
        images = [image for image in images if image.endswith('.png')]

        # for every image in the folder
        for image in images:
            # Load image
            img_path = path + '/' + image
            # print('Processing image: ', img_path)

            # Predict
            results = model(img_path, imgsz = (1920, 1920))
            
            # working with every class obtained
            detected_classes = get_all_classes_obtained(results)
            # Detected classes
            # print('Detected classes: ', detected_classes)

            for result in results:
                for number, name in detected_classes.items():

                    # If there is any object in image
                    if result.masks:

                        IMG_FOLDER = MASK_FOLDER + name + '/' + folder
                        MASK_IMG_PATH = IMG_FOLDER + '/' + image

                        # geting mask from inference
                        get_mask_from_inference(result, MASK_IMG_PATH, IMG_FOLDER, number)



                        # obtaining just the object
                        OBJECT_WITHOUT_BACK_FOLDER = OBJECTS_FOLDER + name + '/' + folder
                        OG_IMAGE_PATH = PROCESSED_FOLDER + folder + '/' + image

                        get_object_from_mask(MASK_IMG_PATH, OG_IMAGE_PATH, image, OBJECT_WITHOUT_BACK_FOLDER)

                        # applying dilatation
                        make_dilatation_2_image(MASK_IMG_PATH)


if __name__ == '__main__':
    print('Starting segmentation...')
    start_time = time.time()
    main()
    end_time = time.time()
    print('Segmentation finished in: ', end_time - start_time)
