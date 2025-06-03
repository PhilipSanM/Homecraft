import os
import flet as ft
import cv2
import numpy as np
from flet_contrib.color_picker import ColorPicker

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return [b, g, r]

def load_image(path):
    try:
        img = cv2.imread(path)
        if img is None:
            with open(path, 'rb') as f:
                bytes = bytearray(f.read())
                img = cv2.imdecode(np.asarray(bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error crítico al cargar imagen: {str(e)}")
        return None

def process_images(input_folder, mask_folder, output_folder, target_color):
    processed_count = 0
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(input_folder, filename)
        mask_path = os.path.join(mask_folder, filename)
        if not os.path.exists(mask_path):
            print(f"❌ Máscara no encontrada para {filename}")
            continue
        img = load_image(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            print(f"❌ Fallo al cargar imagen o máscara: {filename}")
            continue
        if img.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_img)
        target_bgr = np.uint8([[target_color]])
        target_hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)[0][0]
        target_hue = target_hsv[0]
        h[mask > 0] = target_hue
        s_region = s[mask > 0]
        s_boosted = np.clip(np.where(s_region < 30, 100, s_region * 1.8), 50, 255)
        s[mask > 0] = s_boosted.astype(np.uint8)
        modified_hsv = cv2.merge([h, s, v])
        result_img = cv2.cvtColor(modified_hsv, cv2.COLOR_HSV2BGR)
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, result_img)
        processed_count += 1
    return processed_count


def process_color_view(page: ft.Page, appbar, object_name: str):
    result_text=ft.Text("")

    def confirm_processing(e):
        hex_color = ColorPicker()
        print(hex_color)
        try:
            selected_bgr = hex_to_bgr(hex_color)
            print(selected_bgr)
        except:
            print("AAAAaaa")
            page.update()
            return

        input_folder = f"./src/workspace/objects/{object_name}/images"
        mask_folder = f"./src/workspace/mask_room/{object_name}/images"
        output_folder = os.path.join(input_folder, "resultados_coloreados")

        count = process_images(input_folder, mask_folder, output_folder, selected_bgr)
        result_text.value = f"✅ Procesadas {count} imágenes. Resultado en '{output_folder}'"
        result_text.color = "green"
        page.update()

    return ft.View(
        route="/process_color",
        bgcolor="#E5E5E5",
        controls=[
            appbar,
            ft.Column(
                [
                    ft.Text(f"Procesamiento de imágenes para objeto: {object_name}", size=30),
                    ft.Text("Selecciona un color:", size=18),
                    ft.ElevatedButton("Iniciar procesamiento", on_click=confirm_processing),
                    result_text,
                ],
                spacing=20,
                expand=True
            )
        ]
    )