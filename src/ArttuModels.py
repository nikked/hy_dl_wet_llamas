import torch.nn.functional as F
import torch.nn as nn
import torch


class CNN(nn.Module):
    def __init__(self, glove, num_filters=100, filter_sizes=[3, 4, 5], compact_dim=63, dropout=0.0, stride=1):
        super(CNN, self).__init__()

        output_dim = 126
        embedding_dim = len(glove.vectors[0])
        self.dropout = dropout
        self.embedding = nn.Embedding.from_pretrained(
            glove.vectors, freeze=True)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (s, embedding_dim), stride=stride) for s in filter_sizes])
        self.fc1 = nn.Linear(len(filter_sizes) * num_filters, compact_dim)
        self.fc2 = nn.Linear(compact_dim, output_dim)

    def forward(self, X):
        X = self.embedding(X)
        # First convolutions
        conved = [F.relu(conv(X.unsqueeze(1))).squeeze(-1)
                  for conv in self.convs]
        # Pooling
        pooled = [F.max_pool1d(conv, conv.shape[-1]).squeeze(-1)
                  for conv in conved]
        # Fully connected
        X = torch.cat(pooled, dim=-1)
        X = F.dropout(F.relu(self.fc1(X)), p=self.dropout)

        return self.fc2(X)


# Recurrent model
class GRU(nn.Module):
    def __init__(self, glove, hidden_dim=64, num_layers=2, bidirectional=False):
        super(GRU, self).__init__()

        output_dim = 126
        embedding_dim = len(glove.vectors[0])
        self.embedding = nn.Embedding.from_pretrained(
            glove.vectors, freeze=True)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=0.5, bidirectional=bidirectional)
        self.fc = nn.Linear(num_layers * hidden_dim *
                            (2 if bidirectional else 1), output_dim)

    def forward(self, X):
        X = self.embedding(X)
        X, h = self.gru(X)
        h = torch.cat([h[i] for i in range(len(h))], dim=1)
        return self.fc(h)
