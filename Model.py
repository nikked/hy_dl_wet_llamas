import torch.nn.functional as F
import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, glove, num_filters, bottleneck_fc_dim, batch_norm, filter_sizes):
        super(Model, self).__init__()

        output_dim = 126

        embedding_dim = len(glove.vectors[0])
        self.embedding = nn.Embedding.from_pretrained(
            glove.vectors, freeze=True)
        self.conv1 = nn.Conv2d(
            1, num_filters, (filter_sizes[0], embedding_dim))
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(
            1, num_filters, (filter_sizes[1], embedding_dim))
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.conv3 = nn.Conv2d(
            1, num_filters, (filter_sizes[2], embedding_dim))
        self.bn3 = nn.BatchNorm2d(num_filters)
        self.fc1 = nn.Linear(3 * num_filters, bottleneck_fc_dim)
        self.fc2 = nn.Linear(bottleneck_fc_dim, output_dim)
        self.batch_norm = batch_norm

    def forward(self, X):
        X = self.embedding(X)

        X1 = self.conv1(X.unsqueeze(1))
        if self.batch_norm:
            X1 = self.bn1(X1)
        X1 = F.relu(X1.squeeze(-1))

        X2 = self.conv2(X.unsqueeze(1))
        if self.batch_norm:
            X2 = self.bn2(X2)
        X2 = F.relu(X2.squeeze(-1))

        X3 = self.conv3(X.unsqueeze(1))
        if self.batch_norm:
            X3 = self.bn3(X3)
        X3 = F.relu(X3.squeeze(-1))

        X1 = F.max_pool1d(X1, X1.shape[-1]).squeeze(-1)
        X2 = F.max_pool1d(X2, X2.shape[-1]).squeeze(-1)
        X3 = F.max_pool1d(X3, X3.shape[-1]).squeeze(-1)
        X = torch.cat((X1, X2, X3), dim=-1)
        X = self.fc1(X)
        X = F.relu(X)
        X = self.fc2(X)
        return X
