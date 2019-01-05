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


def main():

    models = get_top_model_params()

    for model_name, model_params in models.items():

        txt_length = model_params['txt_length']
        glove = vocab.GloVe(name="6B", dim=model_params['glove_dim'])

        df = load_training_set_as_df(DF_FILEPATH)

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
        train_loader, validation_loader, test_loader = get_loaders(
            df, BATCH_SIZE, NUM_WORKERS, txt_length, glove)

        pkld_file_path = os.path.join(
            MODELS_DIR,
            model_params['train_session_hash'] + '.pkl'
        )

        model.load_state_dict(torch.load(pkld_file_path, map_location='cuda'))

        accuracy, f1_score = measure(model, test_loader)

        print(model_name)
        print(f'Accuracy: {accuracy}')
        print(f'F1 score: {f1_score}')
        print()


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
    main()
