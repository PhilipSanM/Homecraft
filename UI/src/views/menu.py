import flet as ft
import time

def menu_view(page,appbar):
    page.title = "HomeCraft - Menú"
    page.update()
    toolbar_h = 157
    
    imagen_cubo = ft.Image(src='images/cubo.png', width=150, height=150, fit=ft.ImageFit.CONTAIN)
    imagen_f1 = ft.Image(src='images/figuraD2.png', width=150, height=150, fit=ft.ImageFit.CONTAIN)
    imagen_f2 = ft.Image(src='images/figuraD.png', width=150, height=150, fit=ft.ImageFit.CONTAIN)

    contIMG_c = ft.Container(imagen_cubo, left=751, top=130 - toolbar_h, rotate=ft.transform.Rotate(-0.3))
    contIMG_f1 = ft.Container(imagen_f1, left=1713, top=800 - toolbar_h, rotate=ft.transform.Rotate(-0.6))
    contIMG_f2 = ft.Container(imagen_f2, left=122, top=676 - toolbar_h, rotate=ft.transform.Rotate(-0.4))

    
    
    cont_menu = ft.Container(
        content=ft.Text("Elija una opción", size=40),
        alignment= ft.alignment.Alignment(-0.9,-0.9),
        width = 817,
        height = 770,
        left = 551,
        top = 235-toolbar_h,
        bgcolor= "#3A4E7A",
        border_radius=45
        )
    
    
    return ft.View(
        route="/menu",
        bgcolor = "#E5E5E5",
        appbar=appbar,
        controls=[
            ft.Stack([contIMG_c, contIMG_f1, contIMG_f2,cont_menu])
        ]
    )

    