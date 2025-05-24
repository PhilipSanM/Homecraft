import cv2
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def load_images(original_path, processed_path):
    """Carga las imágenes en escala de grises."""
    original = cv2.imread(original_path)
    processed = cv2.imread(processed_path)

    if original is None or processed is None:
        raise FileNotFoundError("Una o ambas rutas no son válidas.")

    # Asegura que tengan el mismo tamaño
    processed = cv2.resize(processed, (original.shape[1], original.shape[0]))

    return original, processed

def calculate_metrics(original, processed):
    """Calcula PSNR y SSIM."""
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    psnr_value = psnr(original_gray, processed_gray)
    ssim_value = ssim(original_gray, processed_gray)

    return psnr_value, ssim_value
IMAGES_COPY = "../processed_room/images_copy"
MASKS_FOLDER = "../objects/background/"
# Ejemplo de uso
avg_psnr = 0
avg_ssim = 0
for filename in os.listdir(IMAGES_COPY):
    original_path = os.path.join(IMAGES_COPY, filename)
    processed_path = os.path.join(MASKS_FOLDER, filename)

    original, processed = load_images(original_path, processed_path)
    psnr_val, ssim_val = calculate_metrics(original, processed)
    avg_ssim += ssim_val
    avg_psnr += psnr_val

    print(f"PSNR: {psnr_val:.2f} dB")
    print(f"SSIM: {ssim_val:.4f}")

print(f"Average PSNR: {avg_psnr/len(os.listdir(IMAGES_COPY)):.2f} dB")
print(f"Average SSIM: {avg_ssim/len(os.listdir(IMAGES_COPY)):.4f}")
