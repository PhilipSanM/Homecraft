import flet as ft
import time
import subprocess
import os


def loading_view(page,appbar):
    page.title = "HomeCraft - Cargando..."
    page.update()
    toolbar_h = 157
    
    imagen_cubo = ft.Image(src='images/cubo.png', width=150, height=150, fit=ft.ImageFit.CONTAIN)
    imagen_f1 = ft.Image(src='images/figuraD2.png', width=150, height=150, fit=ft.ImageFit.CONTAIN)
    imagen_f2 = ft.Image(src='images/figuraD.png', width=150, height=150, fit=ft.ImageFit.CONTAIN)

    contIMG_c = ft.Container(imagen_cubo, left=751, top=130 - toolbar_h, rotate=ft.transform.Rotate(-0.3))
    contIMG_f1 = ft.Container(imagen_f1, left=1713, top=800 - toolbar_h, rotate=ft.transform.Rotate(-0.6))
    contIMG_f2 = ft.Container(imagen_f2, left=122, top=676 - toolbar_h, rotate=ft.transform.Rotate(-0.4))

    

    # Posibles extensiones
    possible_extensions = [".mp4", ".mov", ".avi"]
    workspace_folder = "./src/workspace"

    correct_extension = None  # Inicializa la variable

    # Buscar archivo existente
    for extension in possible_extensions:
        video_path = os.path.join(workspace_folder, f"room{extension}")
        if os.path.exists(video_path):
            correct_extension = extension
            break

    if correct_extension is None:
        raise FileNotFoundError("No se encontró ningún archivo de video en el directorio 'src/worspace/'.")

    # Lista de comandos a ejecutar
    comandos = [
        "docker-compose -f ./src/preprocessing.yaml up -d",
        f'docker exec -it nerfstudio_container bash -c \"ns-process-data video --data nerfstudio/room{correct_extension} --output-dir ./nerfstudio/processed_room\"',
        "docker-compose -f ./src/preprocessing.yaml down"
    ]

    # Ejecutar cada comando en secuencia
    for comando in comandos:
        subprocess.run(comando, shell=True, check=True)

    print("Todos los comandos se ejecutaron correctamente.")
    
    
    btn_ir_menu = ft.ElevatedButton(
        "Ir a menú",
        on_click=lambda e: page.go("/menu"),
        bgcolor="#3A4E7A",
        color="white",
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=20),
            text_style=ft.TextStyle(size=24)
        )
    )
    contBoton = ft.Container(btn_ir_menu,left= 150,top= 150)
    
    
    return ft.View(
        route="/loading",
        bgcolor = "#E5E5E5",
        appbar=appbar,
        controls=[
            ft.Stack([contIMG_c, contIMG_f1, contIMG_f2, contBoton])
        ]
    )

    