from shiny import App, ui

app_ui = ui.page_fluid(
    ui.h2("Hi, WIL Project 48")  
)

def server(input, output, session):
    pass  

app = App(app_ui, server)
