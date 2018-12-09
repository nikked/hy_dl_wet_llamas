from torch.utils.data.dataset import Dataset
import torch


class ReutersDataset(Dataset):
    def __init__(self, df, max_txt_len=1000, glove=None):
        self.max_txt_len = max_txt_len
        self.df = df
        self.topic_codes = initialize_topic_codes()
        self.glove = glove

    def __getitem__(self, index):
        # For now, return simply the headline
        # and corresponding codes. We have to
        # think what we actually want to feed
        # into our model.
        data = self.df.iloc[index]
        h, txt, cs = data.headline, data.text, data.codes
        return self.newsToTensor(h, txt), self.codesToTensor(cs)

    def __len__(self):
        return len(self.df)

    def codesToTensor(self, codes):
        indices = [self.topic_codes.index(c) for c in codes]
        target = torch.zeros(len(self.topic_codes))
        for i in indices:
            target[i] = 1
        return target

    def newsToTensor(self, h, txt):
        # How many characters to take from txt
        max_txt_len = self.max_txt_len
        txt = txt if txt is not None else ""
        txt = txt if len(txt) < max_txt_len else txt[:max_txt_len]
        h = h if h is not None else ""
        feature = h + " " + txt
        return torch.tensor(self.tokenize(feature))

    def tokenize(self, txt):
        words = txt.lower().split()
        tokens = [self.glove.stoi[word]
                  for word in words if word in self.glove.stoi]
        # If token sequence is too short, use this index as padding.
        pad_idx = 400000 - 1
        # This number has to be adjusted to be at least the size of our largest convolution,
        # otherwise the first convolutions will be undefined.
        min_len = 3
        tokens = tokens if len(tokens) >= min_len else tokens + \
            [pad_idx for i in range(min_len - len(tokens))]
        return tokens


def initialize_topic_codes():
    topic_codes = []
    with open("train/topic_codes.txt", "r") as f:
        tc = f.readlines()
        tc = tc[2:]
        for x in tc:
            c, l = x.split("\t", 1)
            topic_codes.append(c.strip())

    return topic_codes
