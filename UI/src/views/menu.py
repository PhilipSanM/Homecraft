import flet as ft
import time

def menu_view(page,appbar):
    page.title = "HomeCraft - Menú"
    page.update()
    toolbar_h = 157
    
    imagen_cubo = ft.Image(src='images/cubo.png', width=150, height=150, fit=ft.ImageFit.CONTAIN)
    imagen_f1 = ft.Image(src='images/figuraD2.png', width=150, height=150, fit=ft.ImageFit.CONTAIN)
    imagen_f2 = ft.Image(src='images/figuraD.png', width=150, height=150, fit=ft.ImageFit.CONTAIN)

    contIMG_c = ft.Container(imagen_cubo, left=751, top=130 - toolbar_h, rotate=ft.Rotate(-0.3))
    contIMG_f1 = ft.Container(imagen_f1, left=1713, top=800 - toolbar_h, rotate=ft.Rotate(-0.6))
    contIMG_f2 = ft.Container(imagen_f2, left=122, top=676 - toolbar_h, rotate=ft.Rotate(-0.4))

    
    
    cont_menu = ft.Container(
        content=ft.Text("Elija una opción", size=40),
        alignment= ft.alignment.Alignment(-0.9,-0.9),
        width = 817,
        height = 650,
        left = 551,
        top = 235-toolbar_h,
        bgcolor= "#3A4E7A",
        border_radius=45
        )
    textBackG = ft.Text("Visualizar fondo", size=34,width=505, text_align="center")
    #textDownObj = ft.Text("Descargar modelo 3D de un objeto", size=34,width=505, text_align="center")
    textVisBack = ft.Text("Visualizar objeto 3D específico", size=34,width=550, text_align="center")
    contBG = ft.Container(content=textBackG, top=460-toolbar_h,left=578)
    #contDO = ft.Container(content=textDownObj, top=551-toolbar_h,left=578)
    contVB = ft.Container(content=textVisBack, top=670-toolbar_h,left=578)
    
    botBackG = ft.ElevatedButton(
        content=ft.Image(src="images/VisBack.png",width=80,height=80),
        bgcolor="#5E83BA",
        height=97,
        width=199,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=25),
        ),
        on_click=lambda e: page.go(f"/visObj?object=background")
    )
    conBotBG= ft.Container(content=botBackG, left=1113, top=450-toolbar_h)
    """
    botDown = ft.ElevatedButton(
        content=ft.Image(src="images/Downl.png",width=80,height=85),
        bgcolor="#5E83BA",
        height=97,
        width=199,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=25),
        ),
        on_click=lambda e: page.go("/loadDL")
    )
    conBotDown= ft.Container(content=botDown, left=1113, top=560-toolbar_h)
    """
    
    botObj = ft.ElevatedButton(
        content=ft.Image(src="images/VisObj.png",width=80,height=80),
        bgcolor="#5E83BA",
        height=97,
        width=199,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=25),
        ),
        on_click=lambda e: page.go("/loadObj")
    )
    conBotObj= ft.Container(content=botObj, left=1113, top=682-toolbar_h)
    
    return ft.View(
        route="/menu",
        bgcolor = "#E5E5E5",
        appbar=appbar,
        controls=[
            ft.Stack([contIMG_c, contIMG_f1, contIMG_f2,cont_menu,contBG, contVB, conBotBG,conBotObj])
        ]
    )