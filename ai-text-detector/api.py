import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse

from static.render import render
from utilities.environment import Environment
from utilities.logging.config import initialize_logging, initialize_logging_middleware
from utilities.utilities import get_uptime
from utilities.exceptions import configure_exception_handlers

import router

import numpy as np
import torch
from model.model import BERTClassifier
from model.data_loader import ConvertRequest
from models.dtos import PredictRequestDto, PredictResponseDto
from loguru import logger
import pandas as pd


SAVE_INPUTS = True
MODEL_WEIGHTS_PATH = './model/trained_models/best.pt'
INPUT_SAVE_PATH = './data/val_data.tsv'
LOG_DISTINATION = './data/logs.log'


text_classifier = BERTClassifier(download_weights=False)
text_classifier.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=torch.device('cpu')))
text_classifier.eval()

request_converter = ConvertRequest(64)


app = FastAPI()

log_format = "<level>{level: <8}</level> | <b>{message}</b> | {file}"
logger.add(LOG_DISTINATION, colorize=False, format=log_format, enqueue=True)


initialize_logging()
initialize_logging_middleware(app)
configure_exception_handlers(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router.router, tags=['AI Text Detector'])


@app.post('/bert/predict', response_model=PredictResponseDto)
def predict(request: PredictRequestDto):

    batch_size = 64
    num_answers = len(request.answers)
    idxs = list(range(0, num_answers, batch_size)) + [num_answers]

    logger.info(f"{num_answers} answers on {len(idxs) - 1} baches")

    output = []
    for pred_num, (i0, i1) in enumerate(zip(idxs[:-1], idxs[1:]), start=1):
        logger.info(f"pred {pred_num}/{len(idxs)-1}")

        model_input = request_converter(request=request.answers[i0:i1])
        preds = text_classifier.predict(**model_input)
        preds = preds.cpu().squeeze().detach().numpy().tolist()
        
        if isinstance(preds, list):
            output.extend(preds)
        else:
            output.append(preds)
            preds = [preds]

    logger.info(f"Completed with {len(output)} preds")
    return PredictResponseDto(
        class_ids=output
    )


@app.post('/save/predict', response_model=PredictResponseDto)
def predict(request: PredictRequestDto):

    if SAVE_INPUTS:
        for idx, text in enumerate(request.answers):
            logger.success(text)
        df = pd.DataFrame(request.answers)
        df.to_csv(INPUT_SAVE_PATH, sep='\t', index=False)
    
    return PredictResponseDto(
        class_ids=len(request.answers) * [0]
    )


@app.post('/guess_true/predict', response_model=PredictResponseDto)
def predict(request: PredictRequestDto):

    return PredictResponseDto(
        class_ids=len(request.answers) * [1]
    )


@app.post('/guess_false/predict', response_model=PredictResponseDto)
def predict(request: PredictRequestDto):

    return PredictResponseDto(
        class_ids=len(request.answers) * [0]
    )


@app.post('/in_method/predict', response_model=PredictResponseDto)
def predict(request: PredictRequestDto):
    
    char = '\n'

    class_ids = len(request.answers) * [None]

    for idx, text in enumerate(request.answers):
        class_ids[idx] = int(char in text)

    return PredictResponseDto(
        class_ids=class_ids
    )


@app.get('/api')
def hello():
    return {
        "service": "ai-text-detector-usecase",
        "uptime": get_uptime()
    }


@app.get('/')
def index():
    return HTMLResponse(
        render(
            'static/index.html',
            host=Environment().HOST_IP,
            port=Environment().CONTAINER_PORT
        )
    )


if __name__ == '__main__':

    uvicorn.run(
        'api:app',
        host=Environment().HOST_IP,
        port=Environment().CONTAINER_PORT
    )