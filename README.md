<p align="center">
    <!-- license badge -->
    <a href="https://github.com/PhilipSanM/Homecraft/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
</p>

<p align="center">
    <!-- logo -->
    <img src="https://github.com/user-attachments/assets/fc36201b-3fcb-4e30-a771-83674ad9e8f3" alt="logo" width="30%">
</p>


<p align="center">
    <i>  A desktop app for creating and editing independent 3D models for interior design and decoration. </i>
</p>



<p align='left'>
    <img src="https://github.com/user-attachments/assets/6c09fb94-9e98-4a04-86a6-b26a66392b72" alt="chair" width="49%">
    <img src="https://github.com/user-attachments/assets/34d37c5c-db56-403d-9824-569ea278631e" alt="background" width="49%">
</p>




## About
HomeCraft is an AI-driven project that allows users to capture a video of their room, extract key frames, segment objects, apply inpainting, and generate a fully editable 3D model of their space. The goal is to enable seamless virtual home redesigns without physically modifying the real-world environment.

## Prerequisites

Before running HomeCraft, ensure you have the following installed:
- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)

Ensure you clone the repository and navigate to its directory:
```bash
git clone https://github.com/your-repo/HomeCraft.git
cd HomeCraft
```

Remember to add a video to workspace folder

```bash
Homecraft/src/workspace/room.mov
```
---

## 1. Preprocessing

### Step 1: Start the Nerfstudio Container
Run the following command to start the preprocessing container:
```bash
docker-compose -f "./src/preprocessing.yaml" up -d
```

### Step 2: Run the Preprocessing Script
Execute the following command inside the container to process the video:
```bash
docker exec -it nerfstudio_container bash -c "ns-process-data video --data nerfstudio/room.mov --output-dir ./nerfstudio/processed_room"
```

### Step 3: Stop and Remove the Nerfstudio Container
After processing, stop and remove the container:
```bash
docker-compose -f "./src/preprocessing.yaml" down
```

## 2. Segmentation

### Step 4: Start the Ultralytics Container
Run the following command to start the segmentation container:
```bash
docker-compose -f "./src/segmentation.yaml" up -d
```

### Step 5: Run the Segmentation Script
Execute the segmentation script to generate the mask folder and objects folder:
```bash
docker exec -it yolo_container bash -c "python ../YOLOv/scripts/segmentation.py"
```

### Step 6: Stop and Remove the Ultralytics Container
After segmentation, stop and remove the container:
```bash
docker-compose -f "./src/segmentation.yaml" down
```

## 3. Stable Diffusion Inpainting

### Step 7: Start the Inpainting Container
Run the following command to start the inpainting container:
```bash
docker-compose -f "./src/inpainting.yaml" up -d
```

### Step 8: Run the Inpainting Script
Generate images of independent objects with the following command:
```bash
docker exec -it SD_container bash -c "python ../SD/scripts/inpainting.py"
```

### Step 9: Stop and Remove the Inpainting Container
After inpainting, stop and remove the container:
```bash
docker-compose -f "./src/inpainting.yaml" down
```

## 4. Postprocessing

### Step 10: Start the Ultralytics Container
Run the following command to start the segmentation container:
```bash
docker-compose -f "./src/segmentation.yaml" up -d
```

### Step 11: Run the Segmentation Script
Execute the postprocessing script to generate the objects for NeRFstudio:
```bash
docker exec -it yolo_container bash -c "python ../YOLOv/scripts/postprocess.py"
```

### Step 12: Stop and Remove the Ultralytics Container
After segmentation, stop and remove the container:
```bash
docker-compose -f "./src/segmentation.yaml" down
```

---

## Visualize Objects

### Step 1: Start the Nerfstudio Container
Run the following command to start the inpainting container:
```bash
docker-compose -f ".\src\preprocessing.yaml" up -d
```

### Step 2: Run the script
Generate images of independent objects with the following command:
```bash
docker exec -it nerfstudio_container bash -c "ns-train nerfacto --data ./nerfstudio/processed_room"
```

### Step 3: Stop and Remove the Nerfstudio Container
After inpainting, stop and remove the container:
```bash
docker-compose -f ".\src\preprocessing.yaml" down
```

---

## Notes
- Ensure all required datasets and models are properly placed in the expected directories before running the scripts.
- Modify paths accordingly if your directory structure is different.
- If any container fails to start, check logs with:
  ```bash
  docker logs <container_name>
  ```
- To remove all stopped containers, run:
  ```bash
  docker system prune -f
  ```

---

## Trabajo a futuro

- UI pantallas faltantes  - EMA
  
- Meter Kmeans / Morfologia     EMI 


- Daniel justificar porq modelos tradicionales vs DNN   EMI
- Gaussplatt HABLAR CON SAULO

- Preguntar macario de arquitectos. (PIPE) Luego mostrarle al rodolfo en revision.
- Conectar los botones de las pantallas con EMA y PIPE   -  Viernes

- Contenedor de MAT y script de inpainting con MAT   OSACR Y PIPE  Manana en la tarde 4🕥

- El script de descargar modelo PIPE
- Contactar sinodales / seguimiento

- Comando de visualizacion trata de jalarlo a la misma UI EMA

- La edicion de color sobre imagenes 2D. PENDIENTE
- Unir Stable diffusion y MAT   DEBATIBLE

- Probar nuevo video del cuarto.  (Lunes)
- Requerimientos de usuarios para subir el video, calidad, tiempo, etc
- Arquitectos:   de a 5 personas mas o menos para saber su opinion o encuestas binarias

- Comparativa de todos los modelos utilizados, el porque y como se quedan en la aplicacion final

### Para la presentacion de TT
- Titulo
- Pequena introduccion de la importancia de porque exploramos este problema, (la MOTIVACION)
- Que son los modelos de inpainting, segmentacion, reconstruccion, el porque y que hacen en la applicacion.
- Introduccion del NERF
- Propuesta de solucion a la problematica (Con los diagramas).
Presentar los objetivos generales y particulares
- La experimentacion de los modelos escogidos y las comparativas.   y puntos especificos del objeto generado (los que son en zoom). (Usando otras herramientas y el tiempo). Reportar los req de computo
- Video demostrativo (el objeto retirado, lo que se obtuvo)
- Conclusiones
- Trabajo a futuro (que consideramos que cambiariamos)


## Ya esta
- Saulo correcciones
- usar bounding box  OSCAR  (Listo)
- Unir mismas mascaras en una imagen  / unir todas las mascaras para hacer el inapinting OSCAR  (Listo)


## YANO
- Stable diffusion hiperparametros OSCAR  YANO
- tener un minimo de imagenes en mascaras por sino eliminar el objetoc  / EMI YANO
  


