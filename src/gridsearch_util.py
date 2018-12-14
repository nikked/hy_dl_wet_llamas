import torch
import torch.nn.functional as F
import pandas as pd


from src.ReutersDataset import ReutersDataset


def load_training_set_as_df(df_filepath):
    df = pd.read_json(df_filepath, compression='xz')
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


def get_loaders(df, batch_size, num_workers, max_txt_len, glove):

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


def train(device, model, epoch, train_loader, optimizer, criterion, train_vector, logs_per_epoch=7):
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


def validate(device, model, test_loader, criterion, loss_vector):
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


def fetch_device(cpu_mode, gpu_no):

    # Fetch device
    if cpu_mode:
        print(f'Using CPU. Too slow for anything serial.')
        return torch.device('cpu')

    elif torch.cuda.is_available():
        print(f'Using GPU. CUDA device #{gpu_no}')
        return torch.device("cuda:{}".format(gpu_no))

    else:
        print("Please use cpu_mode if you don't have cuda GPU available")
        sys.exit(1)
