import flet as ft
from pathlib import Path

def load_download_view(page: ft.Page, appbar):
    page.title = "HomeCraft - Descargar modelos"
    page.scroll = "auto"
    toolbar_h = 157
    selected_objects = []
    imagen_cubo = ft.Image(src='images/cubo.png', width=150, height=150, fit=ft.ImageFit.CONTAIN)
    imagen_f1 = ft.Image(src='images/figuraD2.png', width=150, height=150, fit=ft.ImageFit.CONTAIN)
    imagen_f2 = ft.Image(src='images/figuraD.png', width=150, height=150, fit=ft.ImageFit.CONTAIN)

    contIMG_c = ft.Container(imagen_cubo, left=520, top=200 - toolbar_h, rotate=ft.Rotate(-0.3))
    contIMG_f1 = ft.Container(imagen_f1, left=72, top=720 - toolbar_h, rotate=ft.Rotate(-0.6))
    contIMG_f2 = ft.Container(imagen_f2, left=1600, top=600 - toolbar_h, rotate=ft.Rotate(-0.4))

    ruta_actual = Path(__file__).resolve()
    raiz = ruta_actual.parents[3]
    ruta_objects = raiz / "src" / "workspace" / "objects"

    objetos = []
    if ruta_objects.exists() and ruta_objects.is_dir():
        objetos = [f.name for f in ruta_objects.iterdir() if f.is_dir() and f.name not in ["unknown"]]
    
    else:
        page.go("/menu")
    checkboxes = {}

    def update_selection(e):
        nombre = e.control.data
        if e.control.value:
            if nombre not in selected_objects:
                selected_objects.append(nombre)
        else:
            if nombre in selected_objects:
                selected_objects.remove(nombre)

    lista_objetos_ui = []
    for objeto in objetos:
        cb = ft.Checkbox(data=objeto, on_change=update_selection)
        checkboxes[objeto] = cb

        fila = ft.Row(
            controls=[
                cb,
                ft.Icon(name=ft.Icons.IMAGE, color="#5E83BA"),
                ft.Text(objeto, color="white", expand=True, size=18),
            ],
            alignment="spaceBetween",
            spacing=15
        )
        lista_objetos_ui.append(fila)

    def descargar_modelos(e):
        if selected_objects:
            objetos_param = ",".join(selected_objects)
            print(objetos_param)
            page.go(f"/download_exec?objs={objetos_param}")
            #page.go(f"/descargar?objs={objetos_param}")
        else:
            print("no hay objetos")
            page.open(ft.SnackBar(ft.Text("Selecciona al menos un objeto.", color="white"), bgcolor="red",elevation=100,))
            page.update()

    card_container = ft.Container(
        width=700,
        bgcolor="#2E4172",
        border_radius=20,
        padding=20,
        content=ft.Column([
            ft.Text("Descargar modelos 3D", size=24, color="white", weight="bold"),
            ft.Container(
                height=700,
                content=ft.ListView(controls=lista_objetos_ui, expand=True),
            ),
            ft.Row([
                ft.ElevatedButton(content=ft.Text("Volver", size=18, color="white"), bgcolor="#5E83BA", on_click=lambda e: page.go("/menu")),
                ft.ElevatedButton(content=ft.Row([ft.Icon(ft.Icons.DOWNLOAD, color="white"),ft.Text("Descargar", size=18, color="white")], spacing=5), bgcolor="#5E83BA", on_click=descargar_modelos)
            ], alignment="spaceBetween")
        ])
    )

    return ft.View(
        route="/loadDL",
        appbar=appbar,
        bgcolor="#E5E5E5",
        controls=[
            ft.Stack([
                contIMG_c,
                contIMG_f1,
                contIMG_f2,
                ft.Container(content=card_container, alignment=ft.alignment.center),
            ])
        ]
    )
