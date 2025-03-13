import torch
import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time

# CONSTANTS
YOLO_MODEL = 'yolov8m-seg'


# Workspace from nerfstudio
PROCESSED_FOLDER = '..\\processed_room\\'
IMAGES_FOLDERS = ['images']

# mask folder
MASK_FOLDER = '..\\mask_room\\'

# objects folder
OBJECTS_FOLDER = '..\\objects\\'

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
        features["num_holes"] = sum(1 for i in range(len(hierarchy[0])) if hierarchy[0][i][3] != -1) if hierarchy is not None else 0
        features["angle"] = cv2.fitEllipse(cnt)[2] if len(cnt) >= 5 else 0

    return features

# Función para procesar una carpeta y extraer características
def process_folder(folder):
    data = []
    labels = []
    if os.listdir(folder) and folder == "..\\mask_room\\unknown\\":
        
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
            class_path = class_path  + "\\images" 
            if class_name == 'unknown':
                continue
            elif os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if mask is not None:
                        features = extract_mask_features(mask)
                        data.append(list(features.values()))
                        labels.append(class_name)


        
    return np.array(data), labels

def clasify_unknown(known_class, unknown_clas):
    # Procesar todas las imágenes en carpetas conocidas
    known_data, known_labels = process_folder(known_class)  # Carpeta con imágenes ya clasificadas
    unknown_data, _ = process_folder(unknown_clas)  # Imágenes sin clase

    # Normalizar y reducir la dimensionalidad
    scaler = StandardScaler()
    known_data_scaled = scaler.fit_transform(known_data)
    unknown_data_scaled = scaler.transform(unknown_data)

    pca = PCA(n_components=5)  # Reducir dimensiones para mejorar clustering
    known_data_pca = pca.fit_transform(known_data_scaled)
    unknown_data_pca = pca.transform(unknown_data_scaled)

    # Aplicar K-Means para agrupar
    n_clusters = len(set(known_labels))  # Número de clases conocidas
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(known_data_pca)

    # Predecir clases para imágenes en "unknown"
    unknown_clusters = kmeans.predict(unknown_data_pca)

    # Asignar etiquetas a las imágenes en unknown
    cluster_to_label = {i: known_labels[np.argmax(np.bincount(kmeans.labels_ == i))] for i in range(n_clusters)}
    print(len(unknown_clusters))
    for i, cluster in (enumerate(unknown_clusters)):

        assigned_label = cluster_to_label[cluster]
        #print(f"Imagen {i} asignada a la clase {assigned_label}")
        src_path = os.path.join("..\\mask_room\\unknown\\", os.listdir("..\\mask_room\\unknown\\")[0])
        dest_path = os.path.join(f"..\\mask_room\\{assigned_label}\\images", os.listdir("..\\mask_room\\unknown\\")[0])
        # Mover imagen a carpeta correspondiente
        os.rename(src_path, dest_path)

for class_name in os.listdir(MASK_FOLDER):
        if class_name == 'unknown':
            continue
        print(len(os.listdir(MASK_FOLDER + class_name + '\\' + 'images')))
        if len(os.listdir(MASK_FOLDER + class_name + '\\' + 'images')) == 0:  
            os.rmdir(MASK_FOLDER + class_name + '/' + 'images')
            os.rmdir(MASK_FOLDER + class_name + '/' )
clasify_unknown("..\\mask_room\\", "..\\mask_room\\unknown\\")