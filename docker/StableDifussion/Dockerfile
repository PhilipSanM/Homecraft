# Imagen base con Python y CUDA
FROM pytorch/pytorch:2.5.0-cuda11.8-cudnn9-runtime

# Instalar dependencias del sistema
RUN apt update && apt install -y \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear un entorno de trabajo
WORKDIR /app

# Instalar dependencias de Python
RUN pip install --upgrade pip
RUN pip install diffusers transformers accelerate pillow numpy opencv-python
RUN pip install -U xformers --index-url https://download.pytorch.org/whl/cu118
