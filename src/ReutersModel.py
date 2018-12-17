import torch.nn.functional as F
import torch.nn as nn
import torch


class ReutersModel(nn.Module):
    def __init__(self, glove, num_filters, bottleneck_fc_dim, use_batch_norm, dropout_pctg, filter_sizes, stride):
        super(ReutersModel, self).__init__()

        output_dim = 126

        embedding_dim = len(glove.vectors[0])
        self.embedding = nn.Embedding.from_pretrained(
            glove.vectors, freeze=True)

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters,
                       (filter, embedding_dim),
                       stride=stride)
             for filter in filter_sizes])

        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm2d(num_filters) for filter in filter_sizes])

        self.fc1 = nn.Linear(len(filter_sizes) *
                             num_filters, bottleneck_fc_dim)
        self.fc2 = nn.Linear(bottleneck_fc_dim, output_dim)
        self.use_batch_norm = use_batch_norm

        self.dropout = nn.Dropout(dropout_pctg)

    def forward(self, X):
        X = self.embedding(X)

        conved_layers = []
        for idx, conv in enumerate(self.convs):
            X_conv = conv(X.unsqueeze(1))
            if self.use_batch_norm:

                bn_func = self.batch_norms[idx]
                X_conv = bn_func(X_conv)

            X_conv = F.relu(X_conv.squeeze(-1))
            conved_layers.append(X_conv)

        # Pooling
        pooled = [F.max_pool1d(conv, conv.shape[-1]).squeeze(-1)
                  for conv in conved_layers]

        X = torch.cat((pooled), dim=-1)
        X = self.fc1(X)
        X = F.relu(X)
        X = self.dropout(X)
        X = self.fc2(X)
        return X


class CRNN(nn.Module):
    def __init__(self, glove, num_filters, bottleneck_fc_dim, use_batch_norm,
                 dropout_pctg, filter_sizes, stride,
                 rnn_hidden_size, rnn_num_layers, rnn_bidirectional):
        super(CRNN, self).__init__()

        output_dim = 126

        if rnn_bidirectional:
            bidirectional_multiplier = 2
        else:
            bidirectional_multiplier = 1

        embedding_dim = len(glove.vectors[0])
        self.embedding = nn.Embedding.from_pretrained(
            glove.vectors, freeze=True)

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters,
                       (filter, embedding_dim),
                       stride=stride)
             for filter in filter_sizes])

        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm2d(num_filters) for filter in filter_sizes])

        self.gru = nn.GRU(input_size=embedding_dim,
                          hidden_size=rnn_hidden_size,
                          num_layers=rnn_num_layers,
                          batch_first=True,
                          dropout=dropout_pctg,
                          bidirectional=rnn_bidirectional)

        self.fc1 = nn.Linear(len(filter_sizes) *
                             num_filters + rnn_num_layers * bidirectional_multiplier * rnn_hidden_size, bottleneck_fc_dim)
        self.fc2 = nn.Linear(bottleneck_fc_dim, output_dim)
        self.use_batch_norm = use_batch_norm

        self.dropout = nn.Dropout(dropout_pctg)

    def forward(self, X):
        X = self.embedding(X)

        X_gru, h = self.gru(X)
        h = torch.cat([h[i] for i in range(len(h))], dim=1)

        conved_layers = []
        for idx, conv in enumerate(self.convs):
            X_conv = conv(X.unsqueeze(1))
            if self.use_batch_norm:

                bn_func = self.batch_norms[idx]
                X_conv = bn_func(X_conv)

            X_conv = F.relu(X_conv.squeeze(-1))
            conved_layers.append(X_conv)

        # Pooling
        pooled = [F.max_pool1d(conv, conv.shape[-1]).squeeze(-1)
                  for conv in conved_layers]
        X_conv = torch.cat((pooled), dim=-1)

        X = torch.cat((X_conv, h), dim=1)
        X = self.fc1(X)
        X = F.relu(X)
        X = self.dropout(X)
        X = self.fc2(X)
        return X


class ReutersModelStacked(nn.Module):
    def __init__(self, glove, num_filters, bottleneck_fc_dim, use_batch_norm, dropout_pctg, filter_sizes, stride):
        super(ReutersModelStacked, self).__init__()

        output_dim = 126

        embedding_dim = len(glove.vectors[0])
        self.embedding = nn.Embedding.from_pretrained(
            glove.vectors, freeze=True)

        self.convs = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=num_filters,
                    kernel_size=(2, embedding_dim),
                    stride=stride,
                ),
                nn.BatchNorm2d(num_filters)
            )
                for filter in filter_sizes])

        self.fc1 = nn.Linear(3 * num_filters, bottleneck_fc_dim)
        self.fc2 = nn.Linear(bottleneck_fc_dim, output_dim)

        self.dropout = nn.Dropout(dropout_pctg)

    def forward(self, X):
        X = self.embedding(X)

        conved_layers = []
        for idx, conv in enumerate(self.convs):
            X_conv = conv(X.unsqueeze(1))
            X_conv = F.relu(X_conv.squeeze(-1))
            conved_layers.append(X_conv)

        # Pooling
        pooled = [F.max_pool1d(conv, conv.shape[-1]).squeeze(-1)
                  for conv in conved_layers]

        X = torch.cat((pooled), dim=-1)
        X = self.fc1(X)
        X = F.relu(X)
        X = self.dropout(X)
        X = self.fc2(X)
        return X
