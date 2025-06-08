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
git clone https://github.com/PhilipSanM/Homecraft
cd HomeCraft
```
## Run the app

>[!TIP]
>Create a new virtual environment
```bash
python -m venv .venv
```

Activate the new environment
```bash
.venv\Scripts\activate
```
>[!IMPORTANT]
>Install all required packages
```bash
pip install flet[all]
pip install opencv-python
pip install flet-contrib
```

Now you can run the application
```bash
flet run -d UI
```


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