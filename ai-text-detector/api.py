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

import torch
from model.model import BERTClassifier
from model.data_loader import ConvertRequest
from models.dtos import PredictRequestDto, PredictResponseDto
from loguru import logger


SAVE_INPUTS = False
MODEL_WEIGHTS_PATH = 'model/trained_models/best.pt'

text_classifier = BERTClassifier(download_weights=False)
text_classifier.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
text_classifier.eval()

request_converter = ConvertRequest(100)


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


@app.post('/bert/predict', response_model=PredictResponseDto)
def predict(request: PredictRequestDto):

    if SAVE_INPUTS:
        for idx, text in enumerate(request.answers):
            logger.info(f"{idx:> 4}: {text}")
        
        pass
    
    model_input = request_converter(request=request.answers)
    preds = text_classifier.predict(**model_input)
    preds = preds.cpu().squeeze().detach().numpy().tolist()

    logger.info(', '.join(map(str, preds)))

    return PredictResponseDto(
        class_ids=preds
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