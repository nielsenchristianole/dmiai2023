import os

import tqdm

import torch

from model.model import BERTClassifier
from model.data_loader import ConvertRequest


BATCH_SIZE = 64
CONTEXT_LENGTH = 512
MODEL_WEIGHTS_PATH = './model/trained_models/best.pt'
COM_IN_PATH = './com_in/'
COM_OUT_PATH = './com_out/'
COM_SPLIT = '@@@@@'


request_converter = ConvertRequest(CONTEXT_LENGTH)

text_classifier = BERTClassifier(download_weights=False)
text_classifier.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=torch.device('cpu')))
text_classifier = text_classifier.eval().cuda()


def load_txt_fike(path) -> list[str]:
    with open(path, 'r') as f:
        lines = f.read().split(COM_SPLIT)
    return lines

print('Model is listening for requests...')

while True:

    in_dir = os.listdir(COM_IN_PATH)
    out_dir = os.listdir(COM_OUT_PATH)

    if ('request.txt' not in in_dir) or ('request.txt' in out_dir):
        continue

    answers = load_txt_fike(COM_IN_PATH + 'request.txt')

    print(f'Got {len(answers)} answers')

    num_answers = len(answers)
    idxs = list(range(0, num_answers, BATCH_SIZE)) + [num_answers]

    output = []
    for pred_num, (i0, i1) in enumerate(zip(tqdm.tqdm(idxs[:-1]), idxs[1:]), start=1):

        model_input = request_converter(request=answers[i0:i1])
        preds = text_classifier.predict(**model_input)
        preds = preds.cpu().squeeze().detach().numpy().tolist()
        
        if isinstance(preds, list):
            output.extend(preds)
        else:
            output.append(preds)
            preds = [preds]
    
    class_ids = ','.join(map(str, output))

    print(f'Completed with {len(output)} preds')
    
    with open(COM_OUT_PATH + 'request.txt', 'w') as f:
        f.write(class_ids)


