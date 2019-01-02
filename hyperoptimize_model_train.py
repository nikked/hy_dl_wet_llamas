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

from src.ReutersDataset import ReutersDataset
from src.ReutersModel import ReutersModel, ReutersModelStacked
from src.performance_measures import calculate_f1_score, pAtK
from src.gridsearch_util import load_training_set_as_df, get_loaders, train, validate, fetch_device


DF_FILEPATH = 'train/train.json.xz'
LOG_FP = 'model_stats_hyperopt_181215_optimized.json'
BATCH_SIZE = 256
NUM_WORKERS = 15
EPOCHS = 20
NO_OF_EVALS = 200


def grid_search(cpu_mode=False, gpu_no=0):

    space = {
        "dropout_pctg": hp.uniform("dropout_pctg", 0.01, 0.5),
        "num_filters": hp.quniform("num_filters", 600, 1200, 1.0),
        "bottleneck_fc_dim": hp.quniform("bottleneck_fc_dim", 200, 500, 1.0),
        "glove_dim": hp.choice("glove_dim", [100]),
        "batch_norm": hp.choice("batch_norm", [True, False]),
        "filter_sizes": hp.choice("filter_sizes", [[3, 4, 5], [2, 3, 4], [4, 5, 6], [2, 3, 4, 5], [3, 4, 5, 6]]),
        "txt_length": hp.quniform("txt_length", 500, 1000, 1.0),
        "stride": hp.choice("stride", [1, 2]),
        "gpu_no": gpu_no,
        "cpu_mode": cpu_mode
    }

    trials = Trials()

    best = fmin(fn=train_model, space=space,
                algo=tpe.suggest, max_evals=1000, trials=trials)


def train_model(
        train_params):

    dropout_pctg = round(train_params['dropout_pctg'], 2)
    num_filters = int(train_params['num_filters'])
    bottleneck_fc_dim = int(train_params['bottleneck_fc_dim'])
    glove_dim = train_params['glove_dim']
    batch_norm = train_params['batch_norm']
    filter_sizes = train_params['filter_sizes']
    txt_length = int(train_params['txt_length'])
    stride = train_params['stride']
    gpu_no = train_params['gpu_no']
    cpu_mode = train_params['cpu_mode']

    device = fetch_device(cpu_mode, gpu_no)

    glove = vocab.GloVe(name="6B", dim=glove_dim)

    model = ReutersModel(glove, num_filters, bottleneck_fc_dim,
                         batch_norm, dropout_pctg, filter_sizes, stride)

    model = model.to(device)

    df = load_training_set_as_df(DF_FILEPATH)
    train_loader, valid_loader, test_loader = get_loaders(
        df, BATCH_SIZE, NUM_WORKERS, txt_length, glove)

    # Train params
    train_session_name = f"n_flt:{num_filters}, btl_dim:{bottleneck_fc_dim}, glove:{glove_dim},flt_sz:{filter_sizes},bn:{batch_norm},do_pctg:{dropout_pctg},txt_length:{txt_length},stride:{stride}"
    criterion = nn.BCEWithLogitsLoss()
    parameters = model.parameters()
    optimizer = optim.Adam(parameters)

    try:
        with open(LOG_FP, "r") as file:
            model_stats = json.load(file)
    except Exception as e:
        model_stats = {}

    model_stats[train_session_name] = {
        "dropout_pctg": dropout_pctg,
        "num_filters": num_filters,
        "bottleneck_fc_dim": bottleneck_fc_dim,
        "glove_dim": glove_dim,
        "batch_norm": batch_norm,
        "filter_sizes": filter_sizes,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "epochs": EPOCHS,
        "txt_length": txt_length,
        'train_start': str(datetime.now()),
    }
    loss_vector = None

    try:
        print(f"Starting training: {train_session_name}")
        train_vector, loss_vector = [], []
        for epoch in range(1, EPOCHS + 1):
            print(f'Training epoch no {epoch}')
            train(device, model, epoch, train_loader, optimizer,
                  criterion, train_vector, logs_per_epoch=7)
            validate(device, model, valid_loader, criterion, loss_vector)

            # Make an early quit if the loss is not improving
            if loss_vector.index(min(loss_vector)) < len(loss_vector) - 3:
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
        model_stats[train_session_name]["loss_vector"] = loss_vector

    except Exception as e:
        train_error_message = str(e)
        model_stats[train_session_name]['train_error_message'] = train_error_message

    model_stats[train_session_name]['train_finish'] = str(datetime.now())
    model_stats[train_session_name]["model"] = str(model)

    with open(LOG_FP, "w") as file:
        json.dump(model_stats, file)

    if loss_vector:
        return min(loss_vector)

    else:
        return 1.0


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-g', '--gpu_no', type=int)
    parser.add_argument('-c', '--cpu_mode', action='store_true')

    args = parser.parse_args()

    if args.cpu_mode:
        grid_search(cpu_mode=args.cpu_mode)

    if not args.gpu_no and args.gpu_no != 0:
        print('Please provide GPU # or use CPU')
        sys.exit(1)

    grid_search(gpu_no=args.gpu_no)
