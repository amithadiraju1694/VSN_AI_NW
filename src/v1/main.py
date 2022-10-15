from typing import Union

from fastapi import FastAPI, Request,Form
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="../frontend_templates/")


@app.get("/")
def get_inp_text(request: Request):
    result = "Type few words in any shloka"
    return templates.TemplateResponse('home.html', context={'request': request, 'result': result})
