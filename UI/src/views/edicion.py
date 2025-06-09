import os
import flet as ft
import cv2
import numpy as np
import asyncio
import subprocess
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
    page.title = "HomeCraft - Edición"
    page.update()

    result_text = ft.Text("")
    loading_indicator = ft.ProgressRing(width=40, height=40, stroke_width=4, visible=False, color="#1A1A1A")

    color_picker = ColorPicker("#3A4E7A")
    cont2 = ft.Container(content=color_picker, bgcolor="#2E4172", padding=10, border_radius=15)

    process_btn = ft.ElevatedButton(
        "Iniciar procesamiento", 
        bgcolor="#5E83BA", 
        color="white",
        icon=ft.Icons.FAST_FORWARD_ROUNDED
    )
    backButton = ft.ElevatedButton(
        "Volver",
        bgcolor="#5E83BA",
        color="white",
        on_click=lambda e: page.go("/loadObj"),
        icon=ft.Icons.ARROW_BACK
        )

    async def run_command_async(cmd):
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()

    async def background_processing():
        process_btn.disabled = True
        backButton.disabled = True
        loading_indicator.visible = True
        result_text.value = "Procesando imágenes..."
        result_text.color = "black"
        page.update()

        hex_color = color_picker.color
        try:
            selected_bgr = hex_to_bgr(hex_color)
        except Exception as ex:
            print(f"Error al convertir color: {ex}")
            result_text.value = "❌ Error al procesar el color seleccionado."
            result_text.color = "red"
            loading_indicator.visible = False
            process_btn.disabled = False
            backButton.disabled = False
            page.update()
            return

        input_folder = f"./src/workspace/objects/{object_name}/images"
        mask_folder = f"./src/workspace/mask_room/{object_name}/images"
        output_folder = os.path.join(input_folder, "resultados_coloreados")

        count = await asyncio.to_thread(process_images, input_folder, mask_folder, output_folder, selected_bgr)

        # Comandos Docker
        docker_commands = [
            "docker-compose -f ./src/segmentation.yaml up -d",
            f'docker exec -it yolo_container bash -c "python ../YOLOv/scripts/postprocess_edit.py --object_name {object_name}"',
            "docker-compose -f ./src/segmentation.yaml down"
        ]
        for cmd in docker_commands:
            await run_command_async(cmd)

        result_text.value = f"✅ Procesadas {count} imágenes y ejecutado postprocesado Docker."
        result_text.color = "green"
        loading_indicator.visible = False
        process_btn.disabled = False
        backButton.disabled = False
        page.update()

    def confirm_processing(e):
        page.run_task(background_processing)

    process_btn.on_click = confirm_processing

    background_images = ft.Stack(
        [
            ft.Container(
                content=ft.Image(src='images/cubo.png', width=150, fit=ft.ImageFit.CONTAIN, rotate=ft.Rotate(0.0)),
                alignment=ft.alignment.top_left, expand=True,
            ),
            ft.Container(
                content=ft.Image(src='images/figuraD2.png', width=150, fit=ft.ImageFit.CONTAIN, rotate=ft.Rotate(-0.6)),
                alignment=ft.alignment.center_right, expand=True,
            ),
            ft.Container(
                content=ft.Image(src='images/figuraD.png', width=150, fit=ft.ImageFit.CONTAIN, rotate=ft.Rotate(-0.3)),
                alignment=ft.alignment.bottom_left, expand=True,
            ),
        ]
    )

    foreground_content = ft.Column(
        [
            ft.Text(f"Procesamiento de imágenes para objeto: {object_name}", size=30, color="black"),
            ft.Text("Selecciona un color:", size=18, color="black"),
            cont2,
            ft.Row([process_btn, backButton],alignment=ft.MainAxisAlignment.CENTER),
            loading_indicator,
            result_text,
        ],
        spacing=20,
        expand=True,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
    )

    foreground_container = ft.Container(
        content=foreground_content,
        alignment=ft.alignment.center,
        expand=True,
        padding=30
    )

    return ft.View(
        route="/process_color",
        bgcolor="#E5E5E5",
        controls=[
            appbar,
            ft.Stack(
                [background_images, foreground_container],
                expand=True,
            )
        ]
    )
