import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse

from static.render import render
from utilities.environment import Environment
from utilities.logging.config import (initialize_logging,
                                      initialize_logging_middleware)
from utilities.utilities import get_uptime
from utilities.exceptions import configure_exception_handlers

import router

from PIL import Image
import os
from utils import encode_request, decode_request
from models.dtos import PredictRequestDto, PredictResponseDto
from model.baseline_model import BaselineModel
from models.unet import UNet
import torch


SAVE_INPUT_DATA = False


baseline_model = BaselineModel(label_dir='./data/patients/labels')

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

app.include_router(router.router, tags=['Tumor Segmentation'])


@app.post('/baseline/predict')
def baseline_predict(request: PredictRequestDto):
    
    img = decode_request(request)
    
    if SAVE_INPUT_DATA:
        save_dir = './data/validation_set/'
        idx = len(os.listdir(save_dir))
        save_path = save_dir + f"validation_{idx:0>4}.png"
        Image.fromarray(img).save(save_path)

    pred = baseline_model.predict(img)
    encoded_img = encode_request(pred)

    return PredictResponseDto(
        img=encoded_img
    )

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = 'unet_pet_segmentation_best.pth'

model = UNet(device = DEVICE)
model.load_state_dict(torch.load('./data/weights/unet.pth'))

@app.post('/model/predict')
def predict(request: PredictRequestDto):

    img = decode_request(request)

    pred = model.predict(img)
    encoded_img = encode_request(pred)

    return PredictResponseDto(
        img=encoded_img
    )
    

@app.get('/api')
def hello():
    return {
        "service": "tumor-segmentation-usecase",
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
