from typing import Union

from fastapi import FastAPI, Request,Form, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

import time
from model.pred_nw import predict_next_word
from helpers.model_helps import load_model, validate_inp_text

app = FastAPI()
saved_model = load_model()

templates = Jinja2Templates(directory="templates")
templates.env.autoescape = False
app.mount("/static", 
StaticFiles(directory="static"),
 name="static")  # define the directory for the static files


@app.get("/", response_class=HTMLResponse)
async def get_inp_text( request: Request ):
    return templates.TemplateResponse('home.html', context={'request': request })


@app.post("/", response_class=HTMLResponse)
async def bizl(request: Request):
    user_form = await request.form()

    user_typed = user_form.get("vsn_inp")
    
    st_pred = time.time()
    #TODO: 2. Re-train model to remove ' char from before a word
    #TODO: 3. Loop through results from model and print in new lines in html tags
    shlokas = validate_inp_text(user_typed)

    print("\n Returned shlokas: ", shlokas)


    if len(shlokas) > 0:
        # Call for model predictions here
        predictions = predict_next_word(sentences=shlokas, model = saved_model)

        en_pred = time.time()

        print("\n No. of milli-seconds taken for predictions: ", (en_pred-st_pred)*1000)
        print("\n Predicted next word:" , predictions)
    else:
        predictions = ["NONE"]

    return templates.TemplateResponse('default.html', context={'request': request ,
     'result' : predictions})
