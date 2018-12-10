import torch
import numpy as np
from sklearn import metrics

'''
 Model evaluation
I'll implement for starters the "precision at k" (P@k) evaluation metric. It is defined by
$P@k = \frac{1}{k} \sum_{l=1}^k y_{rank(l)}$,
where $y\in (0,1)^L$ is the true label binary vector and $rank(l)$ is the index of the $l$-th highest label predicted by the system. This metric is averaged over the test set.
'''


def pAtK(device, model, test_loader, k, batch_size):
    p_at_k = 0
    for (X, y) in test_loader:
        X = X.to(device)
        y = y.cpu()
        output = model(X).cpu()
        # Iterate over batch examples
        for batch_idx in range(len(output)):
            k_highest_idx = torch.topk(output[batch_idx], k)[1].numpy()
            # Iterate over predicted labels
            for pred_idx in k_highest_idx:
                p_at_k += y[batch_idx, pred_idx].item()
    p_at_k = p_at_k / (len(test_loader) * batch_size * k)
    return p_at_k


def calculate_f1_score(device, model, test_loader, k, batch_size):

    model.eval()

    true_index_vectors = []
    index_vectors = []

    for (X, y) in test_loader:
        X = X.to(device)
        y = y.to(device)
        output = model(X)

        # Iterate over predicted labels
        for batch_idx in range(len(y)):
            k_highest_idx = torch.topk(output[batch_idx], k)[1].cpu().numpy()

            idx_vector = np.zeros(len(output[batch_idx]))

            for k_high in k_highest_idx:
                idx_vector[k_high] = 1

            index_vectors.append(idx_vector)
            true_index_vectors.append(y[batch_idx].cpu().numpy())

    true_index_vectors = np.array(true_index_vectors)
    index_vectors = np.array(index_vectors)

    f1_score = metrics.f1_score(
        true_index_vectors,
        index_vectors,
        average='micro'
    )

    return f1_score
