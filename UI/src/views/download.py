import flet as ft
import asyncio

class DownloadExecutor(ft.Container):
    def __init__(self, page, objetos_a_descargar):
        super().__init__()
        self.page = page
        self.objetos = objetos_a_descargar
        self.descargados = []

        self.width = 1920
        self.height = 1080
        self.bgcolor = None
        self.padding = 0

        self.text_message = ft.Text("Iniciando descarga...", size=32, color="white")
        self.progress = ft.ProgressRing(width=75, height=75, stroke_width=5, color="#1A1A1A")

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
            margin=ft.margin.only(top=-250)
        )

        self.content = ft.Column(
            [
                ft.Container(content=self.cont, alignment=ft.alignment.center)
            ],
            expand=True,
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER
        )

        self.page.run_task(self.run_downloads)

    async def run_downloads(self):
        for obj in self.objetos:
            self.set_message(f"Descargando {obj}...")
            commands = [
                "docker-compose -f ./src/preprocessing.yaml up -d",
                f'docker exec -it nerfstudio_container bash -c "python ./nerfstudio/scripts/export.py --object_name {obj}"',
                "docker-compose -f ./src/preprocessing.yaml down"
            ]
            await self.run_command_list(commands)
            self.descargados.append(obj)
        self.show_summary()

    def set_message(self, msg):
        self.text_message.value = msg
        self.update()

    def show_summary(self):
        self.cont.content = ft.Column(
            [
                ft.Text("Descarga completada", size=26, color="white", weight="bold"),
                ft.Text("Objetos descargados:", size=20, color="white"),
                ft.ListView(
                    controls=[ft.Text(f"• {obj}", size=18, color="white") for obj in self.descargados],
                    height=200,
                    spacing=10
                ),
                ft.ElevatedButton("Volver al menú", on_click=lambda e: self.page.go("/menu"), bgcolor="#5E83BA", color="white")
            ],
            spacing=20,
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER
        )
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

def download_executor_view(page, appbar, selected_objects):
    page.title = "HomeCraft - Descargando modelos..."
    page.update()

    toolbar_h = 157
    imagen_cubo = ft.Image(src='images/cubo.png', width=150, height=150, fit=ft.ImageFit.CONTAIN)
    imagen_f1 = ft.Image(src='images/figuraD2.png', width=150, height=150, fit=ft.ImageFit.CONTAIN)
    imagen_f2 = ft.Image(src='images/figuraD.png', width=150, height=150, fit=ft.ImageFit.CONTAIN)

    contIMG_c = ft.Container(imagen_cubo, left=751, top=130 - toolbar_h, rotate=ft.Rotate(-0.3))
    contIMG_f1 = ft.Container(imagen_f1, left=1713, top=800 - toolbar_h, rotate=ft.Rotate(-0.6))
    contIMG_f2 = ft.Container(imagen_f2, left=122, top=676 - toolbar_h, rotate=ft.Rotate(-0.4))

    executor = DownloadExecutor(page, selected_objects)

    return ft.View(
        route="/download_exec",
        appbar=appbar,
        bgcolor="#E5E5E5",
        controls=[
            ft.Stack([contIMG_c, contIMG_f1, contIMG_f2, executor])
        ]
    )
