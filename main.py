from fastapi import FastAPI, Request,Form, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

import time
from src.model.pred_nw import predict_next_word
from src.helpers.model_helps import load_model, validate_inp_text
from typing import Union

app = FastAPI()
saved_model = load_model()

templates = Jinja2Templates(directory="./src/templates")
templates.env.autoescape = False
app.mount("/src/static", StaticFiles(directory="./src/static"),name="static")  # define the directory for the static files


@app.get("/", response_class=HTMLResponse)
async def get_inp_text( request: Request ):
    return templates.TemplateResponse('home.html',
     context={'request': request })

@app.get("/useapp", response_class=HTMLResponse)
async def get_usage_page(request: Request):
    return templates.TemplateResponse( 'app_usage.html',
    context={'request': request}
    )

@app.post("/", response_class=HTMLResponse)
async def bizl(request: Request):
    """ Entire business logic works from here."""
    user_form = await request.form()

    user_typed = user_form.get("vsn_inp")
    
    st_pred = time.time()
    
    shlokas = await validate_inp_text(user_typed)

    print("\n Returned shlokas: ", shlokas)


    if len(shlokas) > 0:
        predictions = await predict_next_word(sentences=shlokas,
         model = saved_model)

        en_pred = time.time()

        print("\n No. of milli-seconds taken for predictions: ", (en_pred-st_pred)*1000)
        print("\n Predicted next word:" , predictions)
    else:
        predictions = ["NONE"]

    ret_results = dict(  zip(shlokas, predictions) )

    return templates.TemplateResponse('results.html',
     context={'request': request ,
     'results' : ret_results
     })

#TODO: Replace EOS, with end of sentence