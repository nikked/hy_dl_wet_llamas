from pprint import pprint
import argparse
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

        train_loader, validation_loader, test_loader = get_loaders_with_df(
            glove, model_params)

        accuracy, f1_score = measure(model, test_loader)

        print(model_name)
        print(f'Accuracy: {accuracy}')
        print(f'F1 score: {f1_score}')
        print()


def make_predictions():

    model_name = "{\"batch_norm\": true, \"bottleneck_fc_dim\": 530.0, \"cpu_mode\": false, \"dropout_pctg\": 0.1776, \"epochs\": 20, \"filter_sizes\": [1, 2, 3], \"glove_dim\": 300, \"gpu_no\": 1, \"num_filters\": 762.0, \"rnn_bidirectional\": true, \"rnn_hidden_size\": 652.0, \"rnn_num_layers\": 1.0, \"stride\": 1, \"txt_length\": 775.0}"
    models = get_top_model_params()

    model_params = models[model_name]

    glove = vocab.GloVe(name="6B", dim=model_params['glove_dim'])

    model = load_pretrained_model(model_params, glove)

    train_loader, validation_loader, test_loader = get_loaders_with_df(
        glove, model_params)

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
    with torch.no_grad():
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
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('-p', '--predict', action='store_true')

    args = parser.parse_args()

    if args.evaluate:
        evaluate_f1_scores()

    else:
        make_predictions()

