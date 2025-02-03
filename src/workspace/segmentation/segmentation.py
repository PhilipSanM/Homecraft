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

from ultralytics import YOLO
import torch
import cv2
import os

model = YOLO('yolo11m-seg')

# Workspace from nerfstudio
processed_folder = '../YOLOv/processed_room/'
images_folders = ['images', 'images_2', 'images_4', 'images_8']

# mask folder
mask_folder = '../YOLOv/mask_room/'


for folder in images_folders:
    # Path to the folder
    path = processed_folder + folder
    print('Processing folder: ', path)

    # Find all images in the folder
    images = os.listdir(path)
    images = [image for image in images if image.endswith('.png')]


    for image in images:
        # Load image
        img_path = path + '/' + image
        # print('Processing image: ', img_path)

        # Predict
        results = model(img_path)

        # working with every class obtained
        for result in results:
            yolo_classes = result.names

            for number, name in yolo_classes.items():

                if result.masks:
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

                    # img path

                    img_folder = mask_folder + folder + '/' + name

                    if not os.path.exists(img_folder):
                        os.makedirs(img_folder)

                    img_path = img_folder + '/' + image

                    # print('Saving mask to: ', img_path)

                    # save to file
                    cv2.imwrite(img_path, object_mask.cpu().numpy())




# results = model('../YOLOv/processed_room/images/frame_00001.png')









