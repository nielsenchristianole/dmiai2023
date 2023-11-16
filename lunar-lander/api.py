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

import os
import time
import numpy as np
import torch
from models.dtos import LunarLanderPredictRequestDto, LunarLanderPredictResponseDto
from agent.base_agent import BaselineAgent
from loguru import logger
from ddqn_lunar_lander.ddqn_torch import DoubleQAgent

CHECKPOINT_PATH = 'ddqn_lunar_lander/stats/model_296.h5'

# FILE_LOGS = logger.add('logs/lunar_lander.log', level='SUCCESS')

agent = DoubleQAgent(gamma=0.99, epsilon=0.0, lr=0.0005, mem_size=200000, batch_size=64, epsilon_end=0.01, hidden_size=1028)
agent.load_saved_model(CHECKPOINT_PATH)


app = FastAPI()

initialize_logging()
# initialize_logging_middleware(app)
configure_exception_handlers(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router.router, tags=['Lunar Lander'])

print('Finished loading', os.path.split(CHECKPOINT_PATH)[-1])

FIRST_RUN = True
NEW_RUN = True
RUNNING_TIME = time.time()
TOTAL_TIME = time.time()
TOTAL_RUNS = 0
NUM_CALLS = 0

@app.post('/agent/predict', response_model=LunarLanderPredictResponseDto)
def predict(request: LunarLanderPredictRequestDto):

    global FIRST_RUN
    global NEW_RUN
    global RUNNING_TIME
    global TOTAL_TIME
    global TOTAL_RUNS
    global NUM_CALLS
    NUM_CALLS += 1

    time_now = time.time()
    time_delta = time_now - RUNNING_TIME

    if FIRST_RUN:
        TOTAL_TIME = time_now
        FIRST_RUN = False
    
    if NEW_RUN:
        RUNNING_TIME = time_now
        logger.info(f"Initial state: {request.observation}")
        NEW_RUN = False

    if request.is_terminal:
        TOTAL_RUNS += 1
        logger.success(f"Total Runs: {TOTAL_RUNS}")
        logger.success(f"Total Reward: {request.total_reward}")
        logger.success(f"Time spent: {time_delta}")
        logger.success(f"Total Time spent: {time_now - TOTAL_TIME}")
        logger.success(f"Total API calls: {NUM_CALLS}")
        NEW_RUN = True
        return LunarLanderPredictResponseDto(
            action=0
        )
    
    if time_delta >= 15:
        return LunarLanderPredictResponseDto(
            action=0
        )
    
    state = np.array(request.observation)
    action = agent.choose_action(state)

    return LunarLanderPredictResponseDto(
        action=action.item()
    )


@app.get('/api')
def hello():
    return {
        "service": "lunar-lander-usecase",
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
