import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import json
from datetime import datetime
from torchtext import vocab
import sys
from hyperopt import hp, tpe, fmin, space_eval, Trials
from hyperopt.mongoexp import MongoTrials
from pprint import pprint
import os
import hashlib

from src.ReutersDataset import ReutersDataset
from src.ReutersModel import ReutersModel, ReutersModelStacked, CRNN
from src.performance_measures import calculate_f1_score, pAtK
from src.gridsearch_util import load_training_set_as_df, get_loaders, train, validate, fetch_device


DF_FILEPATH = 'train/train.json.xz'

LOG_DIR = 'log_jsons'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
MODELS_DIR = 'trained_models'
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

LOG_FP = os.path.join(LOG_DIR, f'modelstats_CRNN_{str(datetime.now())}.json')


BATCH_SIZE = 64
NUM_WORKERS = 12
NO_OF_EVALS = 1000


def grid_search(cpu_mode=False, gpu_no=0):

    space = {
        "dropout_pctg": hp.uniform("dropout_pctg", 0.001, 0.5),
        "num_filters": hp.quniform("num_filters", 600, 1500, 1.0),
        "bottleneck_fc_dim": hp.quniform("bottleneck_fc_dim", 200, 600, 1.0),
        "glove_dim": hp.choice("glove_dim", [100, 200, 300]),
        "batch_norm": hp.choice("batch_norm", [True]),
        "filter_sizes": hp.choice("filter_sizes", [[3, 4, 5], [2, 3, 4], [1, 2, 3], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5], [2, 3, 4, 5], [3, 4, 5, 6]]),
        "txt_length": hp.quniform("txt_length", 400, 800, 1.0),
        "stride": hp.choice("stride", [1]),
        "rnn_hidden_size": hp.quniform("rnn_hidden_size", 50, 700, 1.0),
        "rnn_num_layers": hp.quniform("rnn_num_layers", 1, 15, 1.0),
        "rnn_bidirectional": hp.choice("rnn_bidirectional", [True]),
        "epochs": 20,
        "gpu_no": gpu_no,
        "cpu_mode": cpu_mode
    }

    trials = Trials()

    best = fmin(fn=train_model, space=space,
                algo=tpe.suggest, max_evals=NO_OF_EVALS, trials=trials)


def test_grid_search():

    NUM_WORKERS = 0

    space = {
        "dropout_pctg": 0.01,
        "num_filters": 6,
        "bottleneck_fc_dim": 10,
        "glove_dim": 100,
        "batch_norm": False,
        "filter_sizes": [1],
        "txt_length": 20,
        "stride": 1,
        "gpu_no": 0,
        "cpu_mode": True,
        "test_mode": True,
        "rnn_hidden_size": 3,
        "rnn_num_layers": 1,
        "rnn_bidirectional": False,
        "epochs": 1
    }

    train_model(space)


def run_best_models():

    gpu_no = 0

    spaces = [
        {
            "batch_norm": True,
            "bottleneck_fc_dim": 517.0,
            "cpu_mode": False,
            "dropout_pctg": 0.0406,
            "epochs": 20,
            "filter_sizes": [
                1,
                2,
                3,
                4,
                5
            ],
            "glove_dim": 200,
            "gpu_no": gpu_no,
            "num_filters": 1144.0,
            "rnn_bidirectional": True,
            "rnn_hidden_size": 173.0,
            "rnn_num_layers": 1.0,
            "stride": 1,
            "txt_length": 728.0
        },
        {
            "batch_norm": True,
            "bottleneck_fc_dim": 598.0,
            "cpu_mode": False,
            "dropout_pctg": 0.0031,
            "epochs": 20,
            "filter_sizes": [
                1,
                2,
                3,
                4,
                5
            ],
            "glove_dim": 300,
            "gpu_no": gpu_no,
            "num_filters": 796.0,
            "rnn_bidirectional": True,
            "rnn_hidden_size": 60.0,
            "rnn_num_layers": 8.0,
            "stride": 1,
            "txt_length": 784.0
        },
        {
            "batch_norm": True,
            "bottleneck_fc_dim": 597.0,
            "cpu_mode": False,
            "dropout_pctg": 0.0035064775698738504,
            "epochs": 20,
            "filter_sizes": [
                1,
                2,
                3,
                4,
                5
            ],
            "glove_dim": 300,
            "gpu_no": gpu_no,
            "num_filters": 777.0,
            "rnn_bidirectional": True,
            "rnn_hidden_size": 62.0,
            "rnn_num_layers": 8.0,
            "stride": 1,
            "txt_length": 746.0
        }

    ]

    for space in spaces:
        train_model(space)


def train_model(
        train_params):

    if 'test_mode' in train_params:
        test_mode = True
    else:
        test_mode = False

    gpu_no = train_params['gpu_no']
    cpu_mode = train_params['cpu_mode']
    epochs = train_params['epochs']

    dropout_pctg = round(train_params['dropout_pctg'], 2)
    num_filters = int(train_params['num_filters'])
    bottleneck_fc_dim = int(train_params['bottleneck_fc_dim'])
    glove_dim = train_params['glove_dim']
    batch_norm = train_params['batch_norm']
    filter_sizes = train_params['filter_sizes']
    txt_length = int(train_params['txt_length'])
    stride = train_params['stride']

    rnn_hidden_size = int(train_params['rnn_hidden_size'])
    rnn_num_layers = int(train_params['rnn_num_layers'])
    rnn_bidirectional = train_params['rnn_bidirectional']

    try:
        with open(LOG_FP, "r") as file:
            model_stats = json.load(file)
    except Exception as e:
        model_stats = {}

    # Train params
    train_session_name = json.dumps(train_params)

    model_stats[train_session_name] = {
        "dropout_pctg": dropout_pctg,
        "num_filters": num_filters,
        "bottleneck_fc_dim": bottleneck_fc_dim,
        "glove_dim": glove_dim,
        "batch_norm": batch_norm,
        "filter_sizes": filter_sizes,

        "rnn_hidden_size": rnn_hidden_size,
        "rnn_num_layers": rnn_num_layers,
        "rnn_bidirectional": rnn_bidirectional,

        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "epochs": epochs,
        "txt_length": txt_length,
        'train_start': str(datetime.now()),
    }

    valid_vector = None

    try:

        glove = vocab.GloVe(name="6B", dim=glove_dim)

        df = load_training_set_as_df(DF_FILEPATH)
        train_loader, validation_loader, test_loader = get_loaders(
            df, BATCH_SIZE, NUM_WORKERS, txt_length, glove)

        device = fetch_device(cpu_mode, gpu_no)
        model = CRNN(glove, num_filters, bottleneck_fc_dim,
                     batch_norm, dropout_pctg, filter_sizes, stride,
                     rnn_hidden_size, rnn_num_layers, rnn_bidirectional)
        model = model.to(device)

        total_trainable_params = sum(p.numel()
                                     for p in model.parameters() if p.requires_grad)
        model_stats[train_session_name]['no_of_trainable_params'] = total_trainable_params

        criterion = nn.BCEWithLogitsLoss()
        parameters = model.parameters()
        optimizer = optim.Adam(parameters)

        print(f"Starting training: {train_session_name}")
        print(f'No of trainable params in model: {total_trainable_params}')
        train_vector, valid_vector, test_vector = [], [], []

        lowest_loss = 9999
        train_session_hash = hashlib.md5(
            train_session_name.encode()).hexdigest()
        model_path = os.path.join(
            MODELS_DIR,
            train_session_hash + '.pkl')

        model_stats[train_session_name]['train_session_hash'] = train_session_hash

        for epoch in range(1, epochs + 1):
            print(f'Training epoch no {epoch}/{epochs}')
            train(device, model, epoch, train_loader, optimizer,
                  criterion, train_vector, logs_per_epoch=7)
            print('Validating...')
            validate(device, model, validation_loader,
                     criterion, valid_vector, 'Validation')
            validate(device, model, test_loader,
                     criterion, test_vector, 'Test')
            print('\n')

            valid_loss = min(valid_vector)

            if valid_loss < lowest_loss:
                lowest_loss = valid_loss

                print('Saving model to path')
                torch.save(model.state_dict(), model_path)

            # Make an early quit if the loss is not improving
            if valid_vector.index(min(valid_vector)) < len(valid_vector) - 2:
                print('Making an early quit since loss is not improving')
                break

        f1_score_2 = calculate_f1_score(
            device, model, test_loader, 2, BATCH_SIZE)
        f1_score_3 = calculate_f1_score(
            device, model, test_loader, 3, BATCH_SIZE)
        f1_score_4 = calculate_f1_score(
            device, model, test_loader, 4, BATCH_SIZE)

        pAtK_1 = pAtK(device, model, test_loader, 1, BATCH_SIZE)
        pAtK_3 = pAtK(device, model, test_loader, 3, BATCH_SIZE)
        pAtK_5 = pAtK(device, model, test_loader, 5, BATCH_SIZE)

        model_stats[train_session_name]['f1_scores'] = {
            "f1_score_2": f1_score_2,
            "f1_score_3": f1_score_3,
            "f1_score_4": f1_score_4,
        }

        model_stats[train_session_name]['pAtK_scores'] = {
            "pAtK_1": pAtK_1,
            "pAtK_3": pAtK_3,
            "pAtK_5": pAtK_5,
        }

        model_stats[train_session_name]["train_vector"] = train_vector
        model_stats[train_session_name]["valid_vector"] = valid_vector
        model_stats[train_session_name]["test_vector"] = test_vector

    except Exception as e:
        if test_mode:
            pprint(str(e))
            raise Exception(e)
        train_error_message = str(e)
        model_stats[train_session_name]['train_error_message'] = train_error_message

    model_stats[train_session_name]['train_finish'] = str(datetime.now())
    model_stats[train_session_name]["model"] = str(model)

    with open(LOG_FP, "w") as file:
        json.dump(model_stats, file)

    if valid_vector:
        return min(valid_vector)

    else:
        return 1.0


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-g', '--gpu_no', type=int)
    parser.add_argument('-c', '--cpu_mode', action='store_true')
    parser.add_argument('-t', '--test_mode', action='store_true')
    parser.add_argument('-r', '--run_best_models', action='store_true')

    args = parser.parse_args()

    if args.test_mode:
        test_grid_search()

    if args.run_best_models:
        run_best_models()

    elif args.cpu_mode:
        grid_search(cpu_mode=args.cpu_mode)

    elif not args.gpu_no and args.gpu_no != 0:
        print('Please provide GPU # or use CPU')
        sys.exit(1)

    else:
        grid_search(gpu_no=args.gpu_no)
