from typing import Union

from fastapi import FastAPI, Request,Form, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app = FastAPI()
templates = Jinja2Templates(directory="templates")
templates.env.autoescape = False
app.mount("/static", StaticFiles(directory="static"), name="static")  # define the directory for the static files


@app.get("/", response_class=HTMLResponse)
async def get_inp_text( request: Request ):
    return templates.TemplateResponse('home.html', context={'request': request })


@app.post("/", response_class=HTMLResponse)
async def bizl(request: Request):
    user_form = await request.form()

    user_typed = user_form.get("vsn_inp")

    return templates.TemplateResponse('default.html', context={'request': request , 'result' : user_typed})
