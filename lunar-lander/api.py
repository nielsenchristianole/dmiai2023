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

import torch
from models.dtos import LunarLanderPredictRequestDto, LunarLanderPredictResponseDto
from agent.base_agent import BaselineAgent
from loguru import logger


agent = BaselineAgent()

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

app.include_router(router.router, tags=['Lunar Lander'])


@app.post('/predict', response_model=LunarLanderPredictResponseDto)
def predict_closing_price(request: LunarLanderPredictRequestDto):

    reward = f"{request.reward:> 4}"
    is_terminal = f"{request.is_terminal:> 5}"
    total_reward = f"{request.total_reward:> 6}"
    game_ticks = f"{request.game_ticks:> 4}"

    logger.info(f"{reward=}, {is_terminal=}, {total_reward=}, {game_ticks=}")

    if request.is_terminal:
        # should return any action for new game to start
        return LunarLanderPredictResponseDto(
            action=0
        )

    state = torch.FloatTensor(request.observation, device=agent.device)
    action: torch.Tensor = agent.predict(state)

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
