import flet as ft

def home_view(page, appbar):
    page.title = "HomeCraft - Inicio"
    page.update()
    toolbar_h = 157
    video_path = None

    def upload_video(e):
        nonlocal video_path
        if video_path:
            page.go("/loading")
        
        else:
            file_picker.pick_files(allowed_extensions=["mp4", "avi", "mov"])

    def file_selected(e: ft.FilePickerResultEvent):
        nonlocal video_path
        if e.files:
            video_path = e.files[0].path
            upload_button.text = "Subir y procesar"
            page.update()


    
    title = ft.Container(
        ft.Text("Convierte tus imágenes en modelos 3D", size=55, color="#1A1A1A"),
        left=120, top=184 - toolbar_h, shape="rectangle", width=733
    )
    imagen_silla = ft.Image(src='images/silla.gif', width=600, height=500, fit=ft.ImageFit.CONTAIN)
    imagen_cubo = ft.Image(src='images/cubo.png', width=150, height=150, fit=ft.ImageFit.CONTAIN)
    imagen_f1 = ft.Image(src='images/figuraD2.png', width=150, height=150, fit=ft.ImageFit.CONTAIN)
    imagen_f2 = ft.Image(src='images/figuraD.png', width=150, height=150, fit=ft.ImageFit.CONTAIN)

    # Contenedores de imágenes
    cont = ft.Container(imagen_silla, left=210, top=320 - toolbar_h)
    contIMG_c = ft.Container(imagen_cubo, left=1660, top=220 - toolbar_h, rotate=ft.transform.Rotate(0))
    contIMG_f1 = ft.Container(imagen_f1, left=1609, top=900 - toolbar_h, rotate=ft.transform.Rotate(0.2))
    contIMG_f2 = ft.Container(imagen_f2, left=85, top=810 - toolbar_h, rotate=ft.transform.Rotate(0.3))

    upload_button = ft.ElevatedButton(
        "Subir video", 
        on_click=upload_video,
        color="white",
        height=207,
        bgcolor="#3A4E7A",
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=45),
            padding=ft.Padding(20, 10, 20, 10),
            text_style=ft.TextStyle(size=48, font_family="IBM plex mono")
        ),
        width=None
    )
    upload_button_container = ft.Container(upload_button, left=1312, top=589 - toolbar_h, margin=0)

    file_picker = ft.FilePicker(on_result=file_selected)
    page.overlay.append(file_picker)
    return ft.View(
        route="/",
        bgcolor = "#E5E5E5",
        appbar=appbar,
        controls=[
            ft.Stack([title, contIMG_c, contIMG_f1, contIMG_f2, upload_button_container, cont])
        ]
    )
