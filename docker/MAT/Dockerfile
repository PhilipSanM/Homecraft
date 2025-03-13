FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

RUN apt update && apt install -y \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear un entorno de trabajo
WORKDIR /app


# Instalar Git dentro del contenedor
RUN apt-get update && apt-get install -y git

RUN git clone https://github.com/fenglinglwb/MAT.git

RUN pip install --no-cache-dir \
    easydict \
    future \
    matplotlib \
    numpy \
    opencv-python \
    scikit-image \
    scipy \
    click \
    requests \
    tqdm \
    pyspng \
    ninja \
    imageio-ffmpeg==0.4.3 \
    timm \
    psutil \
    scikit-learn


RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

