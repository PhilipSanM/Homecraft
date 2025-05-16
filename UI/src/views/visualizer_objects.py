import flet as ft
import os
import asyncio


class VisualizerExecutor(ft.Container):
    def __init__(self, page, object_name: str):
        super().__init__()
        self.page = page
        self.object_name = object_name
        self.text_message = ft.Text(f"Visualizando {self.object_name}...", size=32, color="white")
        self.progress = ft.ProgressRing(width=75, height=75, stroke_width=5, color="#1A1A1A")

        self.button_open = ft.ElevatedButton(
            text="Abrir Visualizador",
            icon=ft.Icons.OPEN_IN_BROWSER,
            bgcolor="#4B6EAF",
            color="white",
            on_click=lambda e: self.page.launch_url("http://localhost:7007"),
        )

        self.button_back = ft.OutlinedButton(
            text="Volver al menú",
            icon=ft.Icons.ARROW_BACK,
            on_click=self.handle_back,
        )

        self.cont = ft.Container(
            content=ft.Column(
                [self.text_message, self.progress, self.button_open, self.button_back],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=15
            ),
            bgcolor="#3A4E7A",
            border_radius=45,
            padding=ft.Padding(30, 20, 30, 20),
        )

        self.content = ft.Column(
            [ft.Container(content=self.cont, alignment=ft.alignment.center)],
            expand=True,
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER
        )

        self.page.run_task(self.run_visualizer)

    async def run_visualizer(self):
        visualizer_commands = [
            "docker-compose -f ./src/preprocessing.yaml up -d",
            f'docker exec -it nerfstudio_container bash -c "ns-train splatfacto --data ./nerfstudio/objects/{self.object_name} --steps_per_save 100"'
        ]
        await self.run_command_list(visualizer_commands)

    async def handle_back(self, e):
        await self.shutdown_if_running()

        await asyncio.sleep(0.1)  # Pequeña pausa para asegurar liberación de recursos
        self.page.views.clear()
        self.page.go("/loadObj")

    async def shutdown_if_running(self):
        
        check_cmd = 'docker ps --filter "name=nerfstudio_container" --format "{{.Names}}"'
        process = await asyncio.create_subprocess_shell(
            check_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        output, _ = await process.communicate()
        container_name = output.decode().strip()

        if container_name:
            print("Contenedor activo. Apagando...")
            await self.run_command("docker-compose -f ./src/preprocessing.yaml down")
        else:
            print("Contenedor ya estaba apagado.")

    async def run_command_list(self, command_list):
        for cmd in command_list:
            await self.run_command(cmd)

    async def run_command(self, cmd):
        print(f"Ejecutando comando: {cmd}")
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        while True:
            chunk = await process.stdout.read(4096)
            if not chunk:
                break
            print(chunk.decode().strip())
        await process.wait()


def visualizer_objects_view(page, appbar, object_name: str):
    page.title = f"HomeCraft - Visualizando {object_name}"
    page.update()

    toolbar_h = 157

    imagen_cubo = ft.Image(src='images/cubo.png', width=150, height=150, fit=ft.ImageFit.CONTAIN)
    imagen_f1 = ft.Image(src='images/figuraD2.png', width=150, height=150, fit=ft.ImageFit.CONTAIN)
    imagen_f2 = ft.Image(src='images/figuraD.png', width=150, height=150, fit=ft.ImageFit.CONTAIN)

    contIMG_c = ft.Container(imagen_cubo, left=751, top=130 - toolbar_h, rotate=ft.Rotate(-0.3))
    contIMG_f1 = ft.Container(imagen_f1, left=1713, top=800 - toolbar_h, rotate=ft.Rotate(-0.6))
    contIMG_f2 = ft.Container(imagen_f2, left=122, top=676 - toolbar_h, rotate=ft.Rotate(-0.4))

    visualizer_executor = VisualizerExecutor(page, object_name)

    return ft.View(
        route="/visObj",
        bgcolor="#E5E5E5",
        appbar=appbar,
        controls=[
            ft.Container(
                expand=True,
                content=ft.Stack(
                    expand=True,
                    controls=[contIMG_c, contIMG_f1, contIMG_f2, visualizer_executor]
                )
            )
        ]
    ) 