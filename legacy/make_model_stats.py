import json
import matplotlib.pyplot as plt
import numpy as np
import pprint


def plot_lossv_from_model_logs(filepath):

    with open(filepath, 'r') as file:
        model_stats = json.load(file)

    model_validations = {}

    for model in model_stats.keys():
        model_validations[model] = model_stats[model]['loss_vector']

    plt.figure(figsize=(10, 6))

    pprint('asdf')
    for model in model_validations.keys():
        lossv = model_validations[model]
        plt.plot(np.arange(1, len(lossv) + 1), lossv, label=model)

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('loss vs epoch')


plot_lossv_from_model_logs('./model_stats.json')
