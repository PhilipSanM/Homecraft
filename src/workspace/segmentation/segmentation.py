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

model = YOLO('yolov8m-seg')

# Workspace from nerfstudio
processed_folder = '../YOLOv/processed_room/'
images_folders = ['images']

# mask folder
mask_folder = '../YOLOv/mask_room/'

# objects folder
objects_folder = '../YOLOv/objects/'

# deleting folder if exists
if os.path.exists(objects_folder):
    os.system('rm -r ' + objects_folder)


# deleting folder if exists
if os.path.exists(mask_folder):
    os.system('rm -r ' + mask_folder)


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
        results = model(img_path, imgsz = (1920, 1920))
        
        # working with every class obtained
        detected_classes = {}

        for idx, prediction in enumerate(results[0].boxes.xywhn): # change final attribute to desired box format
            cls = int(results[0].boxes.cls[idx].item())
            detected_classes[cls] = results[0].names[cls]


        # Detected classes
        print('Detected classes: ', detected_classes)



        for result in results:
            for number, name in detected_classes.items():

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

                    img_folder = mask_folder + name + '/' + folder

                    if not os.path.exists(img_folder):
                        os.makedirs(img_folder)

                    img_path = img_folder + '/' + image

                    # print('Saving mask to: ', img_path)

                    # save to file
                    cv2.imwrite(img_path, object_mask.cpu().numpy())


                    # obtaining just the object
                    # creating folder
                    if not os.path.exists(objects_folder + name + '/' + folder):
                        os.makedirs(objects_folder + name + '/' + folder)



                    mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    mask = mask.astype('uint8')

                    og_image = cv2.imread(processed_folder + folder + '/' + image)

                    # resizing images
                    height, width = og_image.shape[:2]
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


                    # applying mask
                    masked_image = cv2.bitwise_and(og_image, og_image, mask=mask)

                    # removing black background and saving
                    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2BGRA)
                    masked_image[:, :, 3] = mask

                    # saving masked image
                    masked_file = objects_folder + name + '/' + folder + '/' + image
                    cv2.imwrite(masked_file, masked_image)



                    # dilatation applying
                    kernel = cv2.MORPH_ELLIPSE
                    #kernel = cv2.MORPH_RECT
                    #kernel = cv2.MORPH_CROSS
                    element = cv2.getStructuringElement(kernel, (2 * 5 + 1, 2 * 5 + 1),
                                                        (3, 3))
                    dilatation =  cv2.dilate(cv2.imread(img_path), element, iterations=7)
                    # 3 iterations 
                    #cv2.imwrite(img_path, dilatation)
                    # Convertir la máscara a uint8 (si no lo está)
                    #mask_uint8 = dilatation.cpu().numpy().astype('uint8')

                    # Definir un kernel adecuado. El tamaño dependerá de la separación que presentes.
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # Ajusta el tamaño según sea necesario

                    # Aplicar el cierre morfológico para unir las partes separadas
                    mask_closed = cv2.morphologyEx(dilatation, cv2.MORPH_CLOSE, kernel)

                    # Guardar o usar mask_closed en lugar de la máscara original
                    cv2.imwrite(img_path, mask_closed)








# results = model('../YOLOv/processed_room/images/frame_00001.png')



# predicted_class = {}

# for idx, prediction in enumerate(results[0].boxes.xywhn): # change final attribute to desired box format
#     cls = int(results[0].boxes.cls[idx].item())
#     predicted_class[cls] = results[0].names[cls]


# print(predicted_class)


# YOLO CLASSES
# {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}


