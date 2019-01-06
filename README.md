# Wet Llamas: Text project
> News topic classification from Reuters articles

* `competition_results.txt`: Our test set output made with our best model
* `hyperoptimize_CRNN.py`: the train runner for our best model, the CRNN. This file contains the training process including hyperoptimization, the training itself and logging
* `src/ReutersModel.py`: the Models we tried for this challenge. `class CRNN(nn.Module):` was the one we used for our best models
* `evaluate_trained_models.py`: A script for evaluating the f1 scores and accuracy of trained models. It also made outputs predictions with the `-p` argument. The `competition_results.txt` was made with this
* `evaluate_topic_reasonability.py`: A script that returns the article text and its predicted label names. Used for evaluating whether the model's predictions seem reasonable to a human reader.
* `megalog_llama.json`: Train logs from our best performing models

## Requirements
* Python 3.5+
* pip


## Installation
`pip install -r requirements.txt`


