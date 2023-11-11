import os
from typing import Dict

import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import transformers


MODEL_DIR = 'model/bert_weights/'


class BERTClassifier(torch.nn.Module):

    def __init__(
            self,
            model_dir: str=MODEL_DIR,
            *,
            download_weights: bool=False
        ):
        super(BERTClassifier, self).__init__()
        if download_weights:
            self.bert = transformers.BertModel.from_pretrained(
                "Maltehb/danish-bert-botxo",
                cache_dir=model_dir,
                torchscript=True
            )
        else:
            config = transformers.PretrainedConfig.from_pretrained(
                "Maltehb/danish-bert-botxo",
                cache_dir=model_dir
            )
            self.bert = transformers.BertModel(
                config=config
            )
        self.dense = nn.Sequential(
            nn.Linear(768, 1),
            # nn.ReLU(),
            # nn.Linear(256, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(
            self,
            ids: torch.Tensor,
            mask: torch.Tensor,
            token_type_ids: torch.Tensor
        ) -> torch.Tensor:

        _, x = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )
        x = self.dense(x)
        return x
    
    def _fast_forward(
            self,
            ids: torch.Tensor,
            mask: torch.Tensor,
            token_type_ids: torch.Tensor
        ) -> torch.Tensor:
        if self.training:
            return self.forward(ids=ids, mask=mask, token_type_ids=token_type_ids)
        with torch.no_grad():
            return self.forward(ids=ids, mask=mask, token_type_ids=token_type_ids)
    
    def get_probs(
            self,
            ids: torch.Tensor,
            mask: torch.Tensor,
            token_type_ids: torch.Tensor
        ) -> torch.Tensor:
        x = self._fast_forward(ids=ids, mask=mask, token_type_ids=token_type_ids)
        return self.sigmoid(x)
    
    def predict(
            self,
            ids: torch.Tensor,
            mask: torch.Tensor,
            token_type_ids: torch.Tensor
        ) -> torch.Tensor:
        x = self._fast_forward(ids=ids, mask=mask, token_type_ids=token_type_ids)
        return torch.where(x >= 0, 1, 0)

    def toggle_freeze_bert(self, freeze: bool):
        """
        If freeze is True, set parameters of bert to not require gradients.
        """
        for param in self.bert.parameters():
            param.requires_grad = not freeze


def train_bart(
        epochs: int,
        training_set: DataLoader,
        validation_set: DataLoader,
        model: nn.Module,
        loss_fn: nn.BCEWithLogitsLoss,
        optimizer: torch.optim.Optimizer,
        *,
        verbose: bool=True
    ):

    for epoch in range(epochs):
        
        model.train()
        train_loop = tqdm.tqdm(
            enumerate(training_set),
            total=len(training_set),
            leave=False,
            disable=not verbose
        )
        for batch_num, batch in train_loop:
            
            batch: Dict[str, torch.Tensor]
            label: torch.Tensor = batch.pop('target').unsqueeze(1)
            
            optimizer.zero_grad()
            
            pred = model(
                **batch
            )
            label = label.type_as(pred)

            loss = loss_fn(pred, label)
            loss.backward()
            
            optimizer.step()
            
            pred = np.where(pred >= 0, 1, 0)

            num_correct = sum(1 for a, b in zip(pred, label) if a[0] == b[0])
            num_samples = pred.shape[0]
            accuracy = num_correct / num_samples
    
            train_loop.set_description(f'Epoch={epoch+1}/{epochs}')
            train_loop.set_postfix(loss=loss.item(), acc=accuracy)
        
        model.eval()
        val_loop = tqdm.tqdm(
            enumerate(validation_set),
            total=len(validation_set),
            leave=False,
            disable=not verbose
        )
        for batch_num, batch in val_loop:
            label: torch.Tensor = batch.pop('target').unsqueeze(1)
            
            pred = model(
                **batch
            )
            label = label.type_as(pred)
            loss = loss_fn(pred, label)
            
            pred = np.where(pred >= 0, 1, 0)

            num_correct = sum(1 for a, b in zip(pred, label) if a[0] == b[0])
            num_samples = pred.shape[0]
            accuracy = num_correct / num_samples
    
            val_loop.set_description(f'Epoch={epoch+1}/{epochs}')
            val_loop.set_postfix(loss=loss.item(), acc=accuracy)

            print(f'Epoch={epoch+1}/{epochs} | Loss={loss.item()} | Accuracy={accuracy}')

    return model