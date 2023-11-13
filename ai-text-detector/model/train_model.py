import os
import argparse

import torch

from model.model import BERTClassifier, train_bart
from model.data_loader import BertDataset


if __name__ == '__main__':
    MODEL_DIR = './model/trained_models/'
    os.makedirs(MODEL_DIR, exist_ok=True)

    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--epochs', type=int, default=15)
    parser.add_argument('-o', '--model_name', type=str, default='best.pt')
    parser.add_argument('-d', '--dataset_name', type=str, default='data_version_0.csv')
    parser.add_argument('-c', '--context_length', type=int, default=512)
    parser.add_argument('-do', '--dropout', type=float, default=0.95)

    args = parser.parse_args()

    model = BERTClassifier(
        download_weights=True,
        dropout=args.dropout
    ).cuda()
    model.toggle_freeze_bert(True)

    training_set = BertDataset(
        max_length=args.context_length,
        data_path=f'./data/{args.dataset_name}'
    )
    validation_set = BertDataset(
        max_length=args.context_length,
        data_path=f'./data/data.csv'
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')

    try:
        model = train_bart(
            model=model,
            epochs=args.epochs,
            training_set=training_set.get_dataloader(64, shuffle=True),
            validation_set=validation_set.get_dataloader(len(validation_set), shuffle=False),
            optimizer=optimizer,
            loss_fn=loss_fn,
            label_weight=training_set.calculate_calance_weights()
        )
    except KeyboardInterrupt:
        print('Training interrupted')

    save_path = os.path.join(MODEL_DIR, args.model_name)
    num_models = len(os.listdir(MODEL_DIR))
    if os.path.exists(save_path):
        path, ext = os.path.splitext(save_path)
        save_path = f'{path}_{num_models}{ext}'
    
    torch.save(
        model.state_dict(),
        os.path.join(MODEL_DIR, args.model_name)
    )