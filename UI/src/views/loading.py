import flet as ft
import os
import asyncio
import subprocess

class CommandExecutor(ft.Container):
    def __init__(self, page):
        super().__init__()
        self.page = page
        
        self.width = 1920
        self.height = 1080
        self.bgcolor = None
        self.padding = 0

        # Elementos visuales
        self.text_message = ft.Text("Preparando...", size=32,color="white")
        self.progress = ft.ProgressRing(width=75, height=75, stroke_width=5, color="#1A1A1A")

        # Controles internos posicionados por coordenadas
        self.cont = ft.Container(
            content=ft.Column(
                [
                    self.text_message,
                    self.progress
                ],
                spacing=10,
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER
            ),
            bgcolor="#3A4E7A",
            padding=ft.Padding(left=30, right=30, top=10, bottom=50),
            border_radius=45,
            margin= ft.margin.only(top=-250)
        )

        # Contenedor que se centra vertical y horizontalmente
        self.content = ft.Column(
            [
                ft.Container(
                    content=self.cont,
                    alignment=ft.alignment.center
                )
            ],
            expand=True,
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER
        )



        # Ejecutar comandos en segundo plano
        self.page.run_task(self.run_commands)

    async def run_commands(self):
        workspace_folder = "./src/workspace"
        possible_extensions = [".mp4", ".mov", ".avi"]
        correct_extension = None

        for extension in possible_extensions:
            video_path = os.path.join(workspace_folder, f"room{extension}")
            if os.path.exists(video_path):
                correct_extension = extension
                break

        if correct_extension is None:
            self.set_message("No se encontró un archivo de video.")
            return
        """
        preprocessing_commands = [
            "docker-compose -f ./src/preprocessing.yaml up -d",
            f'docker exec -it nerfstudio_container bash -c "ns-process-data video --data nerfstudio/room{correct_extension} --output-dir ./nerfstudio/processed_room"',
            "docker-compose -f ./src/preprocessing.yaml down"
        ]
        #print("hola")
        segmentation_commands = [
            "docker-compose -f ./src/segmentation.yaml up -d",
            "docker exec -it yolo_container bash -c \"python ../YOLOv/scripts/segmentation.py\"",
            "docker-compose -f ./src/segmentation.yaml down"
        ]
        inpainting_commands = [
            "docker-compose -f ./src/inpainting_mat.yaml up -d",
            "docker exec -it MAT_container bash -c \"python ../MAT/scripts/inpaint_with_mat.py\"",
            "docker-compose -f ./src/inpainting_mat.yaml down"
        ]
        post_processing_commands = [
            "docker-compose -f ./src/segmentation.yaml up -d",
            "docker exec -it yolo_container bash -c \"python ../YOLOv/scripts/postprocess.py\"",
            "docker-compose -f ./src/segmentation.yaml down"
        ]
        
        # Ejecutar y actualizar mensajes
        await self.run_command_list(preprocessing_commands)
        self.set_message("Preprocesamiento completado...")

        await self.run_command_list(segmentation_commands)
        self.set_message("Segmentación completada...")

        await self.run_command_list(inpainting_commands)
        self.set_message("Inpainting completado...")

        await self.run_command_list(post_processing_commands)
        self.set_message("Post-procesamiento listo!")
        """
        self.page.go("/menu")

    def set_message(self, msg):
        self.text_message.value = msg
        self.update()

    async def run_command_list(self, command_list):
        for cmd in command_list:
            await self.run_command(cmd)

    async def run_command(self, comando):
        process = await asyncio.create_subprocess_shell(
            comando,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        while True:
            chunk = await process.stdout.read(4096)
            if not chunk:
                break
            print(chunk.decode().strip())

        await process.wait()

def loading_view(page, appbar):
    page.title = "HomeCraft - Cargando..."
    page.update()

    toolbar_h = 157

    # Elementos gráficos de fondo
    imagen_cubo = ft.Image(src='images/cubo.png', width=150, height=150, fit=ft.ImageFit.CONTAIN)
    imagen_f1 = ft.Image(src='images/figuraD2.png', width=150, height=150, fit=ft.ImageFit.CONTAIN)
    imagen_f2 = ft.Image(src='images/figuraD.png', width=150, height=150, fit=ft.ImageFit.CONTAIN)

    contIMG_c = ft.Container(imagen_cubo, left=751, top=130 - toolbar_h, rotate=ft.Rotate(-0.3))
    contIMG_f1 = ft.Container(imagen_f1, left=1713, top=800 - toolbar_h, rotate=ft.Rotate(-0.6))
    contIMG_f2 = ft.Container(imagen_f2, left=122, top=676 - toolbar_h, rotate=ft.Rotate(-0.4))

    # Agregar el ejecutor de comandos
    command_executor = CommandExecutor(page)

    return ft.View(
        route="/loading",
        bgcolor="#E5E5E5",
        appbar=appbar,
        controls=[
            ft.Stack([contIMG_c, contIMG_f1, contIMG_f2, command_executor])
        ]
    )
