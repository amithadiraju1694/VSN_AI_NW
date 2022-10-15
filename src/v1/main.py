from typing import Union

from fastapi import FastAPI, Request,Form, Response
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="../frontend_templates/")


@app.get("/")
async def get_inp_text( request: Request ):
    return templates.TemplateResponse('home.html', context={'request': request })


@app.post("/")
async def bizl(request: Request):
    user_form = await request.form()

    user_typed = user_form.get("vsn_inp")

    return templates.TemplateResponse('default.html', context={'request': request , 'result' : user_typed})
