import os
from typing import List, Dict

import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import transformers


from models.dtos import PredictRequestDto, PredictResponseDto


DATA_PATH = 'data/data.csv'
MODEL_DIR = 'model/bert_weights/'


class BertDataset(Dataset):
    def __init__(
            self,
            max_length: int = float('inf'),
            data_path: str = DATA_PATH,
            model_dir: str = MODEL_DIR
        ):
        super(BertDataset, self).__init__()
        
        self.data = pd.read_csv(data_path)
        self.max_length = min(
            max([len(text) for text in self.list_data()]),
            max_length
        )
        self.max_length = 100

        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            'Maltehb/danish-bert-botxo',
            cache_dir=model_dir,
            torchscript=True
        )
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        text = self.data.iloc[index, 0]
        
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            padding='max_length',
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.max_length,
            truncation=True
        )        

        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(inputs['token_type_ids'], dtype=torch.long),
            'target': torch.tensor(self.data.iloc[index, 1], dtype=torch.long)
        }
    
    def list_data(self):
        return self.data['text'].values.tolist()

    def get_dataloader(
            self,
            batch_size: int,
            *,
            num_workers: int = max(1, os.cpu_count() - 2),
            shuffle: bool = True,
            kwargs: dict = {}
        ):
        return DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            **kwargs
        )


class ConvertRequest():

    def __init__(
            self,
            max_length: int = float('inf'),
            model_dir: str = MODEL_DIR
        ) -> None:

        self.max_length = max_length
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            'Maltehb/danish-bert-botxo',
            cache_dir=model_dir,
            torchscript=True
        )

    def __call__(self, request: List[str]) -> Dict[str, torch.Tensor]:

        max_length = min(
            max([len(text) for text in request]),
            self.max_length
        )

        num_texts = len(request)
        ids = num_texts * [None]
        mask = num_texts * [None]
        token_type_ids = num_texts * [None]

        for idx, text in enumerate(request):
            encoded = self.tokenizer.encode_plus(
                text,
                None,
                padding='max_length',
                add_special_tokens=True,
                return_attention_mask=True,
                max_length=max_length,
                truncation=True
            )
            ids[idx] = encoded['input_ids']
            mask[idx] = encoded['attention_mask']
            token_type_ids[idx] = encoded['token_type_ids']
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)
        }