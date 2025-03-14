import flet as ft
from views.home import home_view
from views.loading import loading_view


def main(page: ft.Page):
    page.fonts = {
        "IBM plex mono":"/fonts/IBMPlexMono-Regular.ttf"
    }
    page.theme = ft.Theme(font_family="IBM plex mono")
    page.window.width = 1920
    page.window.height = 1080
    page.window.maximized = True
    page.adaptive = True
    toolbar_h = 157
    appbar = ft.AppBar(
        title=ft.Text("HomeCraft", size=48, color="white"),
        bgcolor="#3A4E7A",toolbar_height=toolbar_h-10
    )
    
    def route_change(route):
        page.views.clear()
        if page.route == "/":
            page.views.append(home_view(page,appbar))
        elif page.route == "/loading":
            page.views.append(loading_view(page,appbar))
        page.update()
        
    page.on_route_change = route_change
    page.go(page.route)

ft.app(target=main)
