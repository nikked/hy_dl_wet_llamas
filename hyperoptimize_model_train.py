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


from src.ReutersDataset import ReutersDataset
from src.ReutersModel import ReutersModel
from src.performance_measures import calculate_f1_score, pAtK
from hyperopt import hp, tpe, fmin, space_eval, Trials
from pprint import pprint


DF_FILEPATH = 'train/train.json.xz'
LOG_FP = 'model_stats_hyperopt.json'
BATCH_SIZE = 512
NUM_WORKERS = 15
MAX_TXT_LEN = 500
EPOCHS = 10


def grid_search(cpu_mode=False, gpu_no=0):

    # searchspace = {
    #     "dropout_pctgs": [0.00, 0.36, 0.5],
    #     "num_filters": [50, 100, 150, 200, 300],
    #     "bottleneck_fc_dim": [10, 30, 50, 100],
    #     "glove_dim": [50, 100],
    #     "batch_norm": [True, False],
    #     "filter_sizes": [[3, 4, 5], [1, 3, 5], [1, 4, 7]],
    # }
    searchspace = {
        "dropout_pctg": hp.uniform("dropout_pctg", 0.01, 0.5),
        "num_filters": hp.quniform("num_filters", 50, 500, 1.0),
        "bottleneck_fc_dim": hp.quniform("bottleneck_fc_dim", 10, 200, 1.0),
        "glove_dim": hp.choice("glove_dim", [50, 100]),
        "batch_norm": hp.choice("batch_norm", [True, False]),
        "filter_sizes": hp.choice("filter_sizes", [[3, 4, 5], [1, 3, 5], [1, 4, 7]]),
        "gpu_no": gpu_no,
        "cpu_mode": cpu_mode
    }

    best = fmin(fn=train_model, space=searchspace,
                algo=tpe.suggest, max_evals=100, trials=Trials())

def train_model(
        train_params):

    dropout_pctg = train_params['dropout_pctg']
    num_filters = int(train_params['num_filters'])
    bottleneck_fc_dim = int(train_params['bottleneck_fc_dim'])
    glove_dim = train_params['glove_dim']
    batch_norm = train_params['batch_norm']
    filter_sizes = train_params['filter_sizes']
    gpu_no = train_params['gpu_no']
    cpu_mode = train_params['cpu_mode']

    train_start = str(datetime.now())

    if cpu_mode:
        print(f'Using CPU. Too slow for anything serial.')
        device = torch.device('cpu')

    elif torch.cuda.is_available():
        print(f'Using GPU. CUDA device #{gpu_no}')
        device = torch.device("cuda:{}".format(gpu_no))

    else:
        print("Please use cpu_mode if you don't have cuda GPU available")
        sys.exit(1)

    glove = vocab.GloVe(name="6B", dim=glove_dim)

    model = ReutersModel(glove, num_filters, bottleneck_fc_dim,
                         batch_norm, dropout_pctg, filter_sizes)

    model = model.to(device)

    df = _load_training_set_as_df()
    train_loader, test_loader = _get_loaders(
        df, BATCH_SIZE, NUM_WORKERS, MAX_TXT_LEN, glove)

    # Train params
    train_session_name = f"n_flt:{num_filters}, btl_dim:{bottleneck_fc_dim}, glove:{glove_dim},flt_sz:{filter_sizes},bn:{batch_norm},dd_pctg:{dropout_pctg}"
    criterion = nn.BCEWithLogitsLoss()
    parameters = model.parameters()
    optimizer = optim.Adam(parameters)

    train_error_message = ''

    try:
        print(f"Starting training: {train_session_name}")
        train_vector, loss_vector = [], []
        for epoch in range(1, EPOCHS + 1):
            print(f'Training epoch no {epoch}')
            _train(device, model, epoch, train_loader, optimizer,
                   criterion, train_vector, logs_per_epoch=7)
            _validate(device, model, test_loader, criterion, loss_vector)

        f1_score_2 = calculate_f1_score(
            device, model, test_loader, 2, BATCH_SIZE)
        f1_score_3 = calculate_f1_score(
            device, model, test_loader, 3, BATCH_SIZE)
        f1_score_4 = calculate_f1_score(
            device, model, test_loader, 4, BATCH_SIZE)

        pAtK_1 = pAtK(device, model, test_loader, 1, BATCH_SIZE)
        pAtK_3 = pAtK(device, model, test_loader, 3, BATCH_SIZE)
        pAtK_5 = pAtK(device, model, test_loader, 5, BATCH_SIZE)

    except Exception as e:
        train_error_message = str(e)

    try:
        with open(LOG_FP, "r") as file:
            model_stats = json.load(file)
    except Exception as e:
        model_stats = {}

    model_str = str(model)

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
        "max_txt_len": MAX_TXT_LEN,
        "train_vector": train_vector,
        "loss_vector": loss_vector,
        "model": model_str,
        'train_start': train_start,
        "train_finish": str(datetime.now()),
        "f1_scores": {
            "f1_score_2": f1_score_2,
            "f1_score_3": f1_score_3,
            "f1_score_4": f1_score_4,
        },
        "pAtK_scores": {
            "pAtK_1": pAtK_1,
            "pAtK_3": pAtK_3,
            "pAtK_5": pAtK_5,
        },
        "train_error_message": train_error_message

    }

    with open(LOG_FP, "w") as file:
        json.dump(model_stats, file)

    return f1_score_3


def _load_training_set_as_df():
    df = pd.read_json(DF_FILEPATH, compression='xz')
    df = _clean_df(df)
    return df


def _clean_df(df):
    df.headline.fillna(value=" ", inplace=True)
    df.title.fillna(value=" ", inplace=True)

    re_dict = {"\n": " ",
               "\t": " ",
               "\'s": " \'s",
               "?": " ? ",
               ".": " . ",
               ",": " , ",
               '"': ' " ',
               ":": " : ",
               "*": " * ",
               "(": " ( ", ")": " ) ", "$": " $ ", "/": " / "}
    for to_replace, replacement in re_dict.items():
        df.text = df.text.str.replace(to_replace, replacement)
        df.headline = df.headline.str.replace(to_replace, replacement)

    return df


def _get_loaders(df, batch_size, num_workers, max_txt_len, glove):

    train_set = ReutersDataset(
        df.sample(frac=0.9, random_state=42), max_txt_len=max_txt_len, glove=glove)
    test_set = ReutersDataset(
        df.drop(train_set.df.index), max_txt_len=max_txt_len, glove=glove)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        collate_fn=_pad_collate,
        num_workers=num_workers,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        collate_fn=_pad_collate,
        num_workers=num_workers,
        shuffle=False)

    return train_loader, test_loader


def _pad_collate(batch):
    max_len = max(map(lambda example: len(example[0]), batch))
    xs = [tup[0] for tup in batch]
    xs = torch.stack(list(map(lambda x: _pad_to_length(x, max_len), xs)))
    ys = torch.stack([example[1] for example in batch])
    return xs, ys


def _pad_to_length(tensor, length):
    return F.pad(tensor, (0, length - tensor.shape[0]), 'constant', 0)


def _train(device, model, epoch, train_loader, optimizer, criterion, train_vector, logs_per_epoch=7):
    # Set model to training mode
    model.train()

    train_loss = 0
    # Loop over each batch from the training set
    for batch_idx, (X, y) in enumerate(train_loader):
        X = X.to(device)
        y = y.to(device)
        # Zero gradient buffers
        optimizer.zero_grad()
        # Pass data through the network
        output = model(X)
        # Calculate loss
        loss = criterion(output, y)
        train_loss += loss.item()
        # Backpropagate
        loss.backward()
        # Update weights
        optimizer.step()

    train_loss /= len(train_loader)
    train_vector.append(train_loss)


def _validate(device, model, test_loader, criterion, loss_vector):
    model.eval()
    val_loss = 0
    print('\nValidating...')
    for (X, y) in test_loader:
        X = X.to(device)
        y = y.to(device)
        output = model(X)
        val_loss += criterion(output, y).data.item()

    val_loss /= len(test_loader)
    loss_vector.append(val_loss)

    print('Validation set: Average loss: {:.4f}\n.'.format(val_loss))


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
