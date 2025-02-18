<p align="center">
    <!-- pi package badge -->
    <a href="https://badge.fury.io/py/nerfstudio"><img src="https://badge.fury.io/py/nerfstudio.svg" alt="PyPI version"></a>
    <!-- license badge -->
    <a href="https://github.com/nerfstudio-project/nerfstudio/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
</p>

# Homecraft

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

---

## Preprocessing

### Step 1: Start the Preprocessing Container
Run the following command to start the preprocessing container:
```bash
docker-compose -f "./src/preprocessing.yaml" up -d
```

### Step 2: Run the Preprocessing Script
Execute the following command inside the container to process the video:
```bash
docker exec -it nerfstudio_container bash -c "ns-process-data video --data nerfstudio/room.mov --output-dir ./nerfstudio/processed_room"
```

### Step 3: Stop and Remove the Preprocessing Container
After processing, stop and remove the container:
```bash
docker-compose -f "./src/preprocessing.yaml" down
```

---

## Segmentation

### Step 4: Start the Segmentation Container
Run the following command to start the segmentation container:
```bash
docker-compose -f "./src/segmentation.yaml" up -d
```

### Step 5: Run the Segmentation Script
Execute the segmentation script to generate the mask folder:
```bash
docker exec -it yolo_container bash -c "python ../YOLOv/segmentation/segmentation.py"
```

### Step 6: Stop and Remove the Segmentation Container
After segmentation, stop and remove the container:
```bash
docker-compose -f "./src/segmentation.yaml" down
```

---

## Stable Diffusion Inpainting

### Step 7: Start the Inpainting Container
Run the following command to start the inpainting container:
```bash
docker-compose -f "./src/inpainting.yaml" up -d
```

### Step 8: Run the Inpainting Script
Generate images of independent objects with the following command:
```bash
docker exec -it SD_container bash -c "python ../SD/inpainting/inpainting.py"
```

### Step 9: Stop and Remove the Inpainting Container
After inpainting, stop and remove the container:
```bash
docker-compose -f "./src/inpainting.yaml" down
```

---

## Visualize Object

### Step 10: Start the Nerfstudio Container
Run the following command to start the inpainting container:
```bash
docker-compose -f ".\src\preprocessing.yaml" up -d
```

### Step 8: Run the script
Generate images of independent objects with the following command:
```bash
docker exec -it nerfstudio_container bash -c "ns-train nerfacto --data ./nerfstudio/processed_room"
```

### Step 9: Stop and Remove the nerfstudio Container
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

## Future Enhancements
- Automate workflow with a single script.
- Improve logging and error handling.
- Add support for additional object recognition models.

---

