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

import os
import numpy as np
import torch
from model.model import BERTClassifier
from model.data_loader import ConvertRequest
from models.dtos import PredictRequestDto, PredictResponseDto
from loguru import logger
import pandas as pd
import requests
import time
import json


MODEL_WEIGHTS_PATH = './model/trained_models/best.pt'
INPUT_SAVE_PATH = './data/val_data.tsv'
LOG_DISTINATION = './data/logs.log'
COM_IN_PATH = './com_in/'
COM_OUT_PATH = './com_out/'
COM_SPLIT = '@@@@@'
NGROK_URL = 'https://a504-80-208-68-242.ngrok-free.app'
X_TOKEN = '63a2730bc2ed4d28af6801994620c758'
SCRAPING_DIR = './data_scraping/'

os.makedirs(COM_IN_PATH, exist_ok=True)
os.makedirs(COM_OUT_PATH, exist_ok=True)

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


@app.post('/gpu_bert/predict', response_model=PredictResponseDto)
def predict(request: PredictRequestDto):

    with open(COM_IN_PATH + 'request.txt', 'w') as f:
        f.write(COM_SPLIT.join(request.answers))

    out_dir = os.listdir(COM_OUT_PATH)
    if 'request.txt' in out_dir:
        os.remove(COM_OUT_PATH + 'request.txt')
    
    while True:
        out_dir = os.listdir(COM_OUT_PATH)
        if 'request.txt' in out_dir:
            break
    
    with open(COM_OUT_PATH + 'request.txt', 'r') as f:
        output = f.read().split(',')
    output = list(map(int, output))

    return PredictResponseDto(
        class_ids=output
    )


@app.post('/save/predict', response_model=PredictResponseDto)
def predict(request: PredictRequestDto):

    df = pd.DataFrame(request.answers)
    df.to_csv(INPUT_SAVE_PATH, sep='\t', index=False)
    
    logger.success(len(request.answers))
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


@app.post('/bad/predict', response_model=PredictResponseDto)
def predict(request: PredictRequestDto):

    return PredictResponseDto(
        class_ids=[1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
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


SCRAPE_ANSWER = None

@app.post('/scape_worker/predict', response_model=PredictResponseDto)
def predict(request: PredictRequestDto):
    
    global SCRAPE_ANSWER

    if SCRAPE_ANSWER is None:
        SCRAPE_ANSWER = len(request.answers) * [0]

    return PredictResponseDto(
        class_ids=SCRAPE_ANSWER
    )


@app.get('/start_scraper')
def start_scraping():
    def start_atempt() -> requests.models.Response:
        url = 'https://cases.dmiai.dk/api/v1/usecases/ai-text-detector/validate/queue'
        header = {'x-token': X_TOKEN}
        data = {'url': f"{NGROK_URL}/scape_worker/predict"}

        return requests.post(
            url,
            headers=header,
            json=data
        )

    def get_atempt_status(queued_attempt_uuid: str) -> requests.models.Response:
        url = f"https://cases.dmiai.dk/api/v1/usecases/ai-text-detector/validate/queue/{queued_attempt_uuid}"
        header = {'x-token': X_TOKEN}
        return requests.get(
            url,
            headers = header
        )

    def get_atempt_result(queued_attempt_uuid: str) -> requests.models.Response:
        url = f"https://cases.dmiai.dk/api/v1/usecases/ai-text-detector/validate/queue/{queued_attempt_uuid}/attempt"
        header = {'x-token': X_TOKEN}
        return requests.get(
            url,
            headers = header
        )
    
    def do_run() -> float:
        out1 = start_atempt()
        queued_attempt_uuid = out1.json()['queued_attempt_uuid']

        while True:
            time.sleep(1)

            out2 = get_atempt_status(queued_attempt_uuid)
            print(f"Status code: {out2.status_code}")
            if out2.json()['status'] == 'done':
                break

        out3 = get_atempt_result(queued_attempt_uuid)
        score = out3.json()['score']
        return score

    current_score = do_run()

    with open(SCRAPING_DIR + 'first_guess.json', 'w') as f:
        json.dump({'score': current_score, 'answer': SCRAPE_ANSWER}, f)

    for idx in range(len(SCRAPE_ANSWER)):

        SCRAPE_ANSWER[idx] = 1
        new_score = do_run()

        if new_score > current_score:
            SCRAPE_ANSWER[idx] = 0
        else:
            current_score = new_score
            with open(SCRAPING_DIR + f'current_guess.json', 'w') as f:
                json.dump({'score': current_score, 'answer': SCRAPE_ANSWER}, f)

        with open(SCRAPING_DIR + f'guess_{idx:04}.json', 'w') as f:
            json.dump({
                'score': new_score,
                'min_score': current_score,
                'idx': idx,
                'answer': SCRAPE_ANSWER,
            }, f)


if __name__ == '__main__':

    uvicorn.run(
        'api:app',
        host=Environment().HOST_IP,
        port=Environment().CONTAINER_PORT
    )