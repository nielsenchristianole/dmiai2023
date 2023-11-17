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
NGROK_URL = 'https://0844-80-208-65-234.ngrok-free.app'
X_TOKEN = '63a2730bc2ed4d28af6801994620c758'
SCRAPING_DIR = './data_scraping/'

os.makedirs(COM_IN_PATH, exist_ok=True)
os.makedirs(COM_OUT_PATH, exist_ok=True)


app = FastAPI()


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


@app.post('/gpu_bert/predict', response_model=PredictResponseDto)
def gpu_bert_predict(request: PredictRequestDto):

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


from token_naive_bayes.sklearn import NaiveBayesTFiDF
from sklearn.naive_bayes import MultinomialNB

data_path = './data/labelled_validation_data.tsv'
df = pd.read_csv(data_path, sep='\t')

x_data_raw = df['text'].to_numpy()
y_data = df['is_generated'].to_numpy()

naive_bayes_TFiDF = NaiveBayesTFiDF()
naive_bayes_TFiDF.fit(x_data_raw, y_data)


@app.post('/naive_bayes/predict', response_model=PredictResponseDto)
def naive_bayes_predict(request: PredictRequestDto):

    preds = naive_bayes_TFiDF.predict(np.array(request.answers))

    return PredictResponseDto(
        class_ids=preds.tolist()
    )


@app.post('/save/predict', response_model=PredictResponseDto)
def save_predict(request: PredictRequestDto):

    df = pd.DataFrame(request.answers)
    df.to_csv(INPUT_SAVE_PATH, sep='\t', index=False)
    
    logger.success(len(request.answers))
    return PredictResponseDto(
        class_ids=len(request.answers) * [0]
    )


@app.post('/bad/predict', response_model=PredictResponseDto)
def bad_predict(request: PredictRequestDto):

    answer = (1 - np.array(
        [0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
        dtype=int
    )).tolist()

    return PredictResponseDto(
        class_ids=answer
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


global SCRAPE_ANSWER
SCRAPE_ANSWER: np.ndarray = None

@app.post('/scape_worker/predict', response_model=PredictResponseDto)
def scrape_worker(request: PredictRequestDto):
    
    global SCRAPE_ANSWER

    if SCRAPE_ANSWER is None:
        SCRAPE_ANSWER = np.zeros(len(request.answers), dtype=int)

    return PredictResponseDto(
        class_ids=SCRAPE_ANSWER.tolist()
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

    initial_score = do_run()

    os.makedirs(SCRAPING_DIR, exist_ok=True)

    global SCRAPE_ANSWER

    with open(SCRAPING_DIR + 'first_guess.json', 'w') as f:
        json.dump({'score': initial_score, 'answer': SCRAPE_ANSWER.tolist()}, f, indent=2)

    for idx in range(len(SCRAPE_ANSWER)):
        
        SCRAPE_ANSWER *= 0
        SCRAPE_ANSWER[idx] = 1

        new_score = do_run()

        with open(SCRAPING_DIR + f'guess_{idx:04}.json', 'w') as f:
            json.dump(
                {
                    'is_correct': new_score > initial_score,
                    'score': new_score,
                    'initial_score': initial_score,
                    'idx': idx,
                    'answer': SCRAPE_ANSWER.tolist(),
                },
                f,
                indent=2
            )


if __name__ == '__main__':

    uvicorn.run(
        'api:app',
        host=Environment().HOST_IP,
        port=Environment().CONTAINER_PORT
    )