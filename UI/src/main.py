import flet as ft
from views.home import home_view
from views.loading import loading_view
from views.menu import menu_view
from views.load_objects import load_objects_view
from views.visualizer_objects import visualizer_objects_view
from views.edicion import process_color_view

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
        import urllib.parse

        page.views.clear()
        route_path = page.route
        route_parts = route_path.split("?")

        if route_parts[0] == "/":
            page.views.append(home_view(page, appbar))
        elif route_parts[0] == "/loading":
            page.views.append(loading_view(page, appbar))
        elif route_parts[0] == "/menu":
            page.views.append(menu_view(page, appbar))
        elif route_parts[0] == "/loadObj":
            page.views.append(load_objects_view(page, appbar))
        elif route_parts[0] == "/visObj":
            object_name = "default_object"
            if len(route_parts) > 1:
                params = urllib.parse.parse_qs(route_parts[1])
                object_name = params.get("object", ["default_object"])[0]
            page.views.append(visualizer_objects_view(page, appbar, object_name))
        elif route_parts[0] == "/process_color":
            object_name = "default_object"
            if len(route_parts) > 1:
                params = urllib.parse.parse_qs(route_parts[1])
                object_name = params.get("object", ["default_object"])[0]
            page.views.append(process_color_view(page, appbar, object_name))


        page.update()
        
    page.on_route_change = route_change
    page.go(page.route)

ft.app(target=main)