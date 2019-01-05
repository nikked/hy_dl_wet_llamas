from pprint import pprint
import os
import json
from sklearn import metrics
import torch
from torchtext import vocab
from src.gridsearch_util import load_training_set_as_df, get_loaders
from src.ReutersModel import CRNN


DF_FILEPATH = 'train/train.json.xz'
BATCH_SIZE = 64
NUM_WORKERS = 12
MODELS_DIR = 'trained_models'
FILE_LOGS = 'megalog_llama.json'


def evaluate_f1_scores():

    models = get_top_model_params()

    for model_name, model_params in models.items():

        glove = vocab.GloVe(name="6B", dim=model_params['glove_dim'])

        model = load_pretrained_model(model_params, glove)

        train_loader, validation_loader, test_loader = get_loaders_with_df(glove, model_params)

        accuracy, f1_score = measure(model, test_loader)

        print(model_name)
        print(f'Accuracy: {accuracy}')
        print(f'F1 score: {f1_score}')
        print()


def make_predictions():

    model_name = "{\"batch_norm\": true, \"bottleneck_fc_dim\": 530.0, \"cpu_mode\": false, \"dropout_pctg\": 0.1776, \"epochs\": 20, \"filter_sizes\": [1, 2, 3], \"glove_dim\": 300, \"gpu_no\": 1, \"num_filters\": 762.0, \"rnn_bidirectional\": true, \"rnn_hidden_size\": 652.0, \"rnn_num_layers\": 1.0, \"stride\": 1, \"txt_length\": 775.0}"
    model_params = json.loads("""
    {
    "dropout_pctg": 0.18,
    "num_filters": 762,
    "bottleneck_fc_dim": 530,
    "glove_dim": 300,
    "batch_norm": true,
    "filter_sizes": [
      1,
      2,
      3
    ],
    "rnn_hidden_size": 652,
    "rnn_num_layers": 1,
    "rnn_bidirectional": true,
    "batch_size": 64,
    "num_workers": 12,
    "epochs": 20,
    "txt_length": 775,
    "train_start": "2019-01-05 12:21:32.543911",
    "no_of_trainable_params": 7080642,
    "train_session_hash": "3e5408c8a262291c11430ea31abb6640",
    "f1_scores": {
      "f1_score_2": 0.7213569420012582,
      "f1_score_3": 0.8189189189189189,
      "f1_score_4": 0.7713218182998174
    },
    "pAtK_scores": {
      "pAtK_1": 0.97278118336887,
      "pAtK_3": 0.8445828891257996,
      "pAtK_5": 0.582795842217484
    },
    "train_vector": [
      0.024522422156629642,
      0.016864232985320696,
      0.014754938896747432,
      0.013250884166873967,
      0.01206366778545185
    ],
    "valid_vector": [
      0.01780163169181201,
      0.016153606050374754,
      0.01592947110143512,
      0.016451834844224758,
      0.017033416760846107
    ],
    "test_vector": [
      0.017564620340127807,
      0.01606286458694922,
      0.015831404966292286,
      0.016214815305986764,
      0.016792302983981777
    ],
    "train_finish": "2019-01-05 13:20:57.479166",
    "model": "CRNN(\n  (embedding): Embedding(400000, 300)\n  (convs): ModuleList(\n    (0): Conv2d(1, 762, kernel_size=(1, 300), stride=(1, 1))\n    (1): Conv2d(1, 762, kernel_size=(2, 300), stride=(1, 1))\n    (2): Conv2d(1, 762, kernel_size=(3, 300), stride=(1, 1))\n  )\n  (batch_norms): ModuleList(\n    (0): BatchNorm2d(762, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n    (1): BatchNorm2d(762, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n    (2): BatchNorm2d(762, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  )\n  (gru): GRU(300, 652, batch_first=True, dropout=0.18, bidirectional=True)\n  (fc1): Linear(in_features=3590, out_features=530, bias=True)\n  (fc2): Linear(in_features=530, out_features=126, bias=True)\n  (dropout): Dropout(p=0.18)\n)"
  }
    """)

    glove = vocab.GloVe(name="6B", dim=model_params['glove_dim'])

    model = load_pretrained_model(model_params, glove)

    train_loader, validation_loader, test_loader = get_loaders_with_df(glove, model_params)

    predictions = predict(model, test_loader, device=torch.device('cuda'))

    pprint(predictions)


def get_loaders_with_df(glove, model_params):
    df = load_training_set_as_df(DF_FILEPATH)

    txt_length = model_params['txt_length']
    train_loader, validation_loader, test_loader = get_loaders(
        df, BATCH_SIZE, NUM_WORKERS, txt_length, glove)

    return train_loader, validation_loader, test_loader


def load_pretrained_model(model_params, glove):

    model = CRNN(glove,
                 model_params['num_filters'],
                 model_params['bottleneck_fc_dim'],
                 model_params['batch_norm'],
                 model_params['dropout_pctg'],
                 model_params['filter_sizes'],
                 1,
                 model_params['rnn_hidden_size'],
                 model_params['rnn_num_layers'],
                 model_params['rnn_bidirectional']
                 )

    model = model.to(torch.device('cuda'))

    pkld_file_path = os.path.join(
        MODELS_DIR,
        model_params['train_session_hash'] + '.pkl'
    )

    model.load_state_dict(torch.load(pkld_file_path, map_location='cuda'))

    return model


# Extract predictions from model
def predict(model, data_loader, device=torch.device('cuda')):
    model.eval()
    prediction = torch.tensor([], device=device)
    for (X, _) in data_loader:
        X = X.to(device)
        output = model(X)
        prediction = torch.cat((prediction, output.sigmoid().round()))
    return prediction.cpu().numpy()


# Measure model accuracy and F1-score
def measure(model, data_loader, device=torch.device('cuda')):
    model.eval()
    accuracy = 0
    prediction = torch.tensor([], device=device)
    true_labels = torch.tensor([], device=device)
    with torch.no_grad():
        for (X, y) in data_loader:
            X = X.to(device)
            y = y.to(device)
            output = model(X)
            pred = output.sigmoid().round()
            prediction = torch.cat((prediction, pred))
            true_labels = torch.cat((true_labels, y))
            accuracy += (y * pred).sum().item()
        accuracy = accuracy / len(data_loader.dataset)
        f1_score = metrics.f1_score(true_labels.cpu().numpy(),
                                    prediction.cpu().numpy(),
                                    average='micro')
    return accuracy, f1_score


def get_top_model_params():

    with open(FILE_LOGS, 'r') as file:
        return json.load(file)


if __name__ == '__main__':
    make_predictions()
