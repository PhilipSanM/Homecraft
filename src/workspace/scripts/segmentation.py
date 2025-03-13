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
<<<<<<< HEAD
from distutils.dir_util import copy_tree
import shutil
from PIL import Image
import numpy as np
from PIL import Image
=======
import numpy as np
>>>>>>> a21c6223aa05758647afcb3823d88e93caf9bfc0
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time

# CONSTANTS
YOLO_MODEL = 'yolov8m-seg'


# Workspace from nerfstudio
PROCESSED_FOLDER = '../YOLOv/processed_room/'
IMAGES_COPY = '../YOLOv/processed_room/images_copy/'
IMAGES_FOLDERS = ['images']

# mask folder
MASK_FOLDER = '../YOLOv/mask_room/'

# objects folder
OBJECTS_FOLDER = '../YOLOv/objects/'

# Unkonwn maks folder
UNKNOWN_FOLDER = "../YOLOv/mask_room/unknown/images/"

# FUNCTIONS
def make_dilatation_2_image(image_path):
    # applying dilatation
    kernel = cv2.MORPH_ELLIPSE
    # kernel = cv2.MORPH_RECT
    # kernel = cv2.MORPH_CROSS
    element = cv2.getStructuringElement(kernel, (2 * 5 + 1, 2 * 5 + 1),
                                        (3, 3))
    # Definir un kernel adecuado. El tamaño dependerá de la separación que presentes.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # Ajusta el tamaño según sea necesario

    # Aplicar el cierre morfológico para unir las partes separadas
    mask_closed = cv2.morphologyEx(cv2.imread(image_path), cv2.MORPH_CLOSE, kernel)
    dilatation =  cv2.dilate(mask_closed, element, iterations=7)
    # Guardar o usar mask_closed en lugar de la máscara original

    # 3 iterations 
    cv2.imwrite(image_path, dilatation)



def get_object_from_mask(mask_path, og_image_path, image, masked_file_path):

    # if not folder exists, create it
    if not os.path.exists(masked_file_path):
        os.makedirs(masked_file_path)
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

def replace_masks_with_filled_bbox(unknown_folder):
    files = os.listdir(unknown_folder)

    if not files:
        return
    
    for file in files:
        img_path = os.path.join(unknown_folder, file)
        
        # Cargar imagen en escala de grises
        mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        
        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        
        # Obtener la bounding box más grande
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        
        # Crear una nueva imagen en negro del mismo tamaño
        new_mask = np.zeros_like(mask)  # Imagen en negro del mismo tamaño
        
        # Dibujar un rectángulo blanco relleno
        cv2.rectangle(new_mask, (x-15, y-15), (x + w + 15, y + h + 15), 255, thickness=-1)  # Blanco relleno (-1)

        # Guardar la nueva máscara en la misma ruta
        cv2.imwrite(img_path, new_mask)

# Llamar a la función


# def get_mask_from_inference(result, mask_img_path, img_folder, number):
#     # get array results
#     masks = result.masks.data
#     boxes = result.boxes.data

#     # extract classes
#     clss = boxes[:, 5]
#     # get indices of results where class is 0 (people in COCO)
#     object_indices = torch.where(clss == number)
#     # use these indices to extract the relevant masks
#     object_masks = masks[object_indices]

#     # scale for visualizing results
#     object_mask = torch.any(object_masks, dim=0).int() * 255

#     # make folder if not exists
#     if not os.path.exists(img_folder):
#         os.makedirs(img_folder)
    
#     # save to file
#     cv2.imwrite(mask_img_path, object_mask.cpu().numpy())

        
def get_mask_from_inference(result, mask_img_path, img_folder, number, conf_threshold=0.7, unknown_folder = UNKNOWN_FOLDER):
    # Crear carpeta si no existe
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    # Obtener resultados
    masks = result.masks.data
    boxes = result.boxes.data
    confs = result.boxes.conf  # Confianza de cada detección
    clss = boxes[:, 5]  # Clases detectadas

    # Filtrar detecciones de la clase deseada
    object_indices = torch.where(clss == number)

    for idx in object_indices[0]:
        conf = confs[idx].item()
        object_mask = masks[idx].int().cpu().numpy() * 255
        if conf >= conf_threshold: 
            cv2.imwrite(mask_img_path, object_mask)
        else :        
            mask_img_path = os.path.join(unknown_folder, f"{mask_img_path.split('/')[-1]}")
            cv2.imwrite(mask_img_path, object_mask)


def get_features_from_inference(result, mask_img_path, img_folder, number):

    # Si la máscara no es binaria, se aplica un umbral para convertirla
    _, binary = cv2.threshold(cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE), 127, 255, cv2.THRESH_BINARY)

    # Encontrar los contornos en la máscara
    contornos, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contornos:
        raise ValueError("No se encontró ningún contorno en la máscara.")

    # Seleccionar el contorno más grande (o aplicar otro criterio de selección)
    cnt = max(contornos, key=cv2.contourArea)

    # Calcular el área
    area = cv2.contourArea(cnt)

    # Calcular el perímetro (longitud del contorno)
    perimetro = cv2.arcLength(cnt, True)

    # Calcular el círculo mínimo que encierra el contorno: obtiene el centro y el radio
    (x, y), radio = cv2.minEnclosingCircle(cnt)
    centro_circulo = (int(x), int(y))
    radio = int(radio)

    # Calcular el centroide utilizando momentos
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0

    return area, perimetro, centro_circulo, radio, (cx, cy)




# Función para extraer características de la máscara

def extract_mask_features(mask):
    features = {}
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        cnt = contours[0]
        features["area"] = cv2.countNonZero(mask)
        features["perimeter"] = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        features["bounding_box_ratio"] = w / h if h > 0 else 0
        M = cv2.moments(cnt)
        features["centroid_x"] = (M["m10"] / M["m00"]) if M["m00"] != 0 else 0
        features["centroid_y"] = (M["m01"] / M["m00"]) if M["m00"] != 0 else 0
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        features["radius"] = radius
        features["extent"] = features["area"] / (w * h) if w * h > 0 else 0
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        features["solidity"] = features["area"] / hull_area if hull_area > 0 else 0

    return features

# Función para procesar una carpeta y extraer características
def process_folder(folder):
    data = []
    labels = []
    if os.listdir(folder) and folder == UNKNOWN_FOLDER:
        
        class_path = folder
        if os.path.isdir(class_path):
            
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                
                mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if mask is not None:
                    
                    features = extract_mask_features(mask)
                    
                    data.append(list(features.values()))
                    
    else:
        for class_name in os.listdir(folder):  # Carpeta con subcarpetas de clases
            
            class_path = os.path.join(folder, class_name)
            class_path = class_path  + "/images" 
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if mask is not None:
                        features = extract_mask_features(mask)
                        data.append(list(features.values()))
                        labels.append(class_name)


        
    return np.array(data), labels

def clasify_unknown(known_class, unknown_class, threshold=0.85):
    # Procesar todas las imágenes en carpetas conocidas
    known_data, known_labels = process_folder(known_class)  # Carpeta con imágenes ya clasificadas
    unknown_data, _ = process_folder(unknown_class)  # Imágenes sin clase

    # Normalizar los datos
    scaler = StandardScaler()
    known_data_scaled = scaler.fit_transform(known_data)
    unknown_data_scaled = scaler.transform(unknown_data)

    # Reducir dimensionalidad con PCA
    pca = PCA(n_components=4)
    known_data_pca = pca.fit_transform(known_data_scaled)
    unknown_data_pca = pca.transform(unknown_data_scaled)

    # Aplicar KMeans
    n_clusters = len(set(known_labels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(known_data_pca)

    # Predecir clases para imágenes desconocidas
    unknown_clusters = kmeans.predict(unknown_data_pca)
    cluster_to_label = {i: known_labels[np.argmax(np.bincount(kmeans.labels_ == i))] for i in range(n_clusters)}

    # Calcular distancias para threshold
    distances = np.linalg.norm(unknown_data_pca - kmeans.cluster_centers_[unknown_clusters], axis=1)
    threshold_distance = np.percentile(distances, threshold * 90)
    j = 0
    
    for i, cluster in enumerate(unknown_clusters):
        if len(unknown_clusters) <= j:
            j = 0
        assigned_label = cluster_to_label[cluster]
        src_path = os.path.join(UNKNOWN_FOLDER, os.listdir(UNKNOWN_FOLDER)[j])

        if distances[i] > threshold_distance:
            assigned_label = "unknown"
        if assigned_label == "unknown": 

            dest_path = os.path.join(f"../YOLOv/mask_room/unknown/images", os.listdir(UNKNOWN_FOLDER)[j])

        else :
            dest_path = os.path.join(f"../YOLOv/mask_room/{assigned_label}/images", os.listdir(UNKNOWN_FOLDER)[j])
            unknown_clusters = np.delete(unknown_clusters, j)
            j += 1

        # Mover la imagen a la carpeta correspondiente
        os.rename(src_path, dest_path)

def get_all_classes_obtained(results):
    detected_classes = {}

    for idx, prediction in enumerate(results[0].boxes.xywhn): # change final attribute to desired box format
        cls = int(results[0].boxes.cls[idx].item())
        detected_classes[cls] = results[0].names[cls]

    return detected_classes

<<<<<<< HEAD
def resize_images_in_folder(folder, output_folder, size=(512, 512)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for image_name in os.listdir(folder):
        image_path = os.path.join(folder, image_name)
        image = Image.open(image_path)
        image = image.resize(size)
        output_path = os.path.join(output_folder, image_name)
        image.save(output_path)
        #print(f"Resized image saved as {output_path}")
# Example Usage
=======
def image_move():
>>>>>>> a21c6223aa05758647afcb3823d88e93caf9bfc0

    masks_folder = os.path.join(MASK_FOLDER, "masks")

    # Crear la carpeta masks si no existe
    os.makedirs(masks_folder, exist_ok=True)
    masks_folder = masks_folder + '/' + 'images/'
    os.makedirs(masks_folder, exist_ok=True)



    # Recorrer las imágenes en mask_room
    for class_name in os.listdir(MASK_FOLDER):
        if class_name == "masks":
            continue
        class_path = os.path.join(MASK_FOLDER, class_name)+ '/images/'
        # Recorrer las imágenes en la carpeta de la clase
        for filename in os.listdir(class_path):

            file_path = os.path.join(class_path, filename)
            os.rename(file_path, os.path.join(masks_folder, filename))
    # Obtener los nombres de archivos en processed_room (sin extensión)
    processed_files = {f for f in os.listdir(masks_folder) if os.path.isfile(os.path.join(masks_folder, f))}

    # Recorrer las imágenes procesadas
    processed_images = PROCESSED_FOLDER + 'images/'

    for filename in os.listdir(processed_images):        
        # elmiminar archivos que no se encuentren en processed_room
        if filename not in processed_files:
            os.remove(os.path.join(processed_images, filename))



        #print(f"Resized image saved as {output_path}")

def image_move():

    masks_folder = os.path.join(MASK_FOLDER, "masks")

    # Crear la carpeta masks si no existe
    os.makedirs(masks_folder, exist_ok=True)
    masks_folder = masks_folder + '/' + 'images/'
    os.makedirs(masks_folder, exist_ok=True)



    # Recorrer las imágenes en mask_room
    for class_name in os.listdir(MASK_FOLDER):
        if class_name == "masks":
            continue
        class_path = os.path.join(MASK_FOLDER, class_name)+ '/images/'
        # Recorrer las imágenes en la carpeta de la clase
        for filename in os.listdir(class_path):

            file_path = os.path.join(class_path, filename)
            def binarize_and_invert(image_path, output_path, threshold=128):
                """
                1. Converts the image to black and white (0 and 255).
                2. Inverts the colors: Black (0) becomes White (255) and vice versa.
                """
                # Open image and convert to grayscale
                image = Image.open(image_path).convert("L")  # Convert to grayscale

                # Convert image to NumPy array for processing
                img_array = np.array(image)

                # Apply binarization (Thresholding)
                img_array = np.where(img_array > threshold, 255, 0)

                # Invert colors (0 ↔ 255)
                img_array = 255 - img_array

                # Convert array back to an image
                processed_image = Image.fromarray(img_array.astype(np.uint8))

                # Save the processed image
                processed_image.save(output_path)
            binarize_and_invert(file_path, os.path.join(masks_folder, filename))
            #os.rename(file_path, os.path.join(masks_folder, filename))
    # Obtener los nombres de archivos en processed_room (sin extensión)
    processed_files = {f for f in os.listdir(masks_folder) if os.path.isfile(os.path.join(masks_folder, f))}
    # Recorrer las imágenes procesadas
    processed_images = PROCESSED_FOLDER + 'images/'
    if not os.path.exists(IMAGES_COPY):
            os.makedirs(IMAGES_COPY)
    copy_tree(processed_images, IMAGES_COPY)
    for filename in os.listdir(IMAGES_COPY):        
        #crear una copia de la imagen 
        
        # elmiminar archivos que no se encuentren en processed_room
        if filename not in processed_files:
            os.remove(os.path.join(processed_images, filename))

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
        
        if not os.path.exists("../YOLOv/mask_room/unknown/"):
            os.makedirs("../YOLOv/mask_room/unknown/")
            os.makedirs("../YOLOv/mask_room/unknown/"+ folder + '/')

        # Find all images in the folder
        images = os.listdir(path)
        images = [image for image in images if image.endswith('.png')]

        # for every image in the folder
        for image in images:
            # Load image
            img_path = path + '/' + image
            # Predict
            results = model(img_path, imgsz = (1920, 1920))
        
            # working with every class obtained
            detected_classes = get_all_classes_obtained(results)
            # Detected classes
            for result in results:
                for number, name in detected_classes.items():
                    # If there is any object in image
                    if result.masks:

                        IMG_FOLDER = MASK_FOLDER + name + '/' + folder
                        MASK_IMG_PATH = IMG_FOLDER + '/' + image

                        # geting mask from inference
                        get_mask_from_inference(result, MASK_IMG_PATH, IMG_FOLDER, number)
                        
    for class_name in os.listdir(MASK_FOLDER):
        if len(os.listdir(MASK_FOLDER + class_name + '/' + folder + '/' )) == 0:  
            os.system('rm -r ' + MASK_FOLDER + class_name + '/' + folder + '/')
            os.system('rm -r ' + MASK_FOLDER + class_name)
    clasify_unknown(MASK_FOLDER, UNKNOWN_FOLDER)
    folder = 'images'
    for name in os.listdir(MASK_FOLDER):
        class_path = MASK_FOLDER + name + '/' + folder + '/'
        for image in os.listdir(class_path):
<<<<<<< HEAD

            image_path = os.path.join(class_path, image)
            # obtaining just the object
            OBJECT_WITHOUT_BACK_FOLDER = OBJECTS_FOLDER + name + '/' + folder
            OG_IMAGE_PATH = PROCESSED_FOLDER + folder + '/' + image

            get_object_from_mask(image_path, OG_IMAGE_PATH, image, OBJECT_WITHOUT_BACK_FOLDER)

            # applying dilatation
            make_dilatation_2_image(image_path)

       
    replace_masks_with_filled_bbox(UNKNOWN_FOLDER)
    image_move()
    masks_folder = os.path.join(MASK_FOLDER, "masks")
    masks_folder = masks_folder + '/' + 'images/'
    # Resize masks
    folder = masks_folder
    output_folder = masks_folder
    resize_images_in_folder(folder, output_folder)
    #Resize images in folder
    folder = IMAGES_COPY
    output_folder = IMAGES_COPY
    resize_images_in_folder(folder, output_folder)
=======
>>>>>>> a21c6223aa05758647afcb3823d88e93caf9bfc0

            image_path = os.path.join(class_path, image)
            # obtaining just the object
            OBJECT_WITHOUT_BACK_FOLDER = OBJECTS_FOLDER + name + '/' + folder
            OG_IMAGE_PATH = PROCESSED_FOLDER + folder + '/' + image

            get_object_from_mask(image_path, OG_IMAGE_PATH, image, OBJECT_WITHOUT_BACK_FOLDER)

<<<<<<< HEAD
=======
            # applying dilatation
            make_dilatation_2_image(image_path)

       
    replace_masks_with_filled_bbox(UNKNOWN_FOLDER)
    image_move()


>>>>>>> a21c6223aa05758647afcb3823d88e93caf9bfc0

if __name__ == '__main__':
    print('Starting segmentation...')
    start_time = time.time()
    main()
    end_time = time.time()
    print('Segmentation finished in: ', end_time - start_time)
