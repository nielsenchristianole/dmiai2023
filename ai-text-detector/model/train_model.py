import os
import argparse

import torch

from model.model import BERTClassifier, train_bart
from model.data_loader import BertDataset


if __name__ == '__main__':
    MODEL_DIR = './model/trained_models/'
    os.makedirs(MODEL_DIR, exist_ok=True)

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--model_name', type=str, default='best.pt')

    args = parser.parse_args()

    model = BERTClassifier(download_weights=True)
    model.toggle_freeze_bert(True)

    dataset = BertDataset(max_length=100)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    model = train_bart(
        model=model,
        epochs=args.epochs,
        training_set=dataset.get_dataloader(len(dataset), shuffle=True),
        validation_set=dataset.get_dataloader(len(dataset), shuffle=False),
        optimizer=optimizer,
        loss_fn=loss_fn
    )

    torch.save(
        model.state_dict(),
        os.path.join(MODEL_DIR, args.model_name)
    )