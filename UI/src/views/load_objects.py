import flet as ft
from pathlib import Path

def load_objects_view(page: ft.Page, appbar):
    page.title = "HomeCraft - Objetos"
    page.scroll = "auto"
    toolbar_h = 157

    # Imágenes decorativas de fondo
    imagen_cubo = ft.Image(src='images/cubo.png', width=150, height=150, fit=ft.ImageFit.CONTAIN)
    imagen_f1 = ft.Image(src='images/figuraD2.png', width=150, height=150, fit=ft.ImageFit.CONTAIN)
    imagen_f2 = ft.Image(src='images/figuraD.png', width=150, height=150, fit=ft.ImageFit.CONTAIN)

    contIMG_c = ft.Container(imagen_cubo, left=520, top=200 - toolbar_h, rotate=ft.Rotate(-0.3))
    contIMG_f1 = ft.Container(imagen_f1, left=72, top=720 - toolbar_h, rotate=ft.Rotate(-0.6))
    contIMG_f2 = ft.Container(imagen_f2, left=1600, top=600 - toolbar_h, rotate=ft.Rotate(-0.4))

    # Obtener ruta a "src/workspace/objects"
    ruta_actual = Path(__file__).resolve()
    raiz = ruta_actual.parents[3]
    ruta_objects = raiz / "src" / "workspace" / "objects"

    # Leer carpetas válidas
    objetos = []
    if ruta_objects.exists() and ruta_objects.is_dir():
        objetos = [
            f.name for f in ruta_objects.iterdir()
            if f.is_dir() and f.name not in ["background", "unknown"]
        ]

    # Contenedor principal
    card_container = ft.Container(
        width=700,
        bgcolor="#2E4172",
        border_radius=20,
        padding=20,
        content=ft.Column([
            ft.Text("Objetos en la escena", size=24, color="white", weight="bold"),
            ft.Container(
                height=700,
                bgcolor="#2E4172",
                content=ft.ListView(
                    controls=[
                        ft.Row(
                            controls=[
                                ft.Icon(name=ft.Icons.IMAGE, color="#5E83BA"),
                                ft.Text(objeto, color="white", expand=True, size=18),
                                ft.ElevatedButton(
                                    "Ver",
                                    icon=ft.Icons.PLAY_ARROW,
                                    icon_color="#5E83BA",
                                    on_click=lambda e, obj=objeto: page.go(f"/visObj?object={obj}")
                                ),
                                ft.ElevatedButton(
                                    "Editar",
                                    icon=ft.Icons.PALETTE,
                                    icon_color="#5E83BA",
                                    on_click= lambda e, obj = objeto: page.go(f"/process_color?object={obj}")
                                )
                            ],
                            alignment="spaceBetween",
                            spacing=10
                        )
                        for objeto in objetos
                    ],
                    expand=True,
                    
                ),
            ),
            ft.ElevatedButton("Volver", bgcolor="#5E83BA", color="white",on_click= lambda e: page.go("/menu"))
        ])
    )

    cont_card = ft.Container(content=card_container, alignment=ft.alignment.center)


    return ft.View(
        route="/loadObj",
        appbar=appbar,
        bgcolor="#E5E5E5",
        controls=[
            ft.Stack(
                controls=[
                    contIMG_c,
                    contIMG_f1,
                    contIMG_f2,
                    cont_card,
                ]
            )
        ],
        scroll=ft.ScrollMode.AUTO
    )