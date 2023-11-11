

import torch
import torch.nn as nn

# from models.dtos import LunarLanderPredictRequestDto, LunarLanderPredictResponseDto


class BaselineAgent(nn.Module):

    def __init__(self, device: str=None) -> None:
        super().__init__()

        self.device = torch.device(device or 'cuda:0') if (device or torch.cuda.is_available()) else torch.device('cpu')

    def predict(
        self,
        state: torch.Tensor
    ) -> torch.Tensor:
        """
        Do the prediction

        Arguments
        ---------
        state: a torch tensor on self.device

        Output
        ------
        action: a 0d torch.Tensor on self.device with dtype=torch.int32
        """
        return torch.tensor(0, device=self.device, dtype=torch.int32)

