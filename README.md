## News article classification with Deep Learning

This repository consists of the code for a text classification competition organized at the University of Helsinki. The competition was part of a Deep Learning course. Our team won the challenge :)

The task consists of classifying news articles from Reuters to topic codes. It is a multi-label classification problem: each article can correspond to none, one or multiple topics. We solved this problem with a deep neural network architecture we call CRNN, a hybrid model of a convolutional and a recurrent network.

Our award-winning was quite fancy. For tokenization, we used pre-trained GloVe-embeddings. We optimized the model's various hyperparameters with hyperopt. We used a powerful NVIDIA Quadro P4000 GPU for training. As a result, we achieved 97% accuracy for one label prediction.

For more details, please refer to the [our full report of the competition entry.](https://github.com/nikked/hy_dl_wet_llamas/blob/master/reuters_article_classification.pdf)



### Repository content
* `hyperoptimize_CRNN.py`: The main CLI of the project that runs the neural network. This file facilitates the whole training process including hyperoptimization, the training itself and logging
* `src/ReutersModel.py`: This file holds the Models we tried for this challenge. `class CRNN(nn.Module):` was the one we used for our best models
* `src/topic_codes.txt`: The topic codes we were trying to predict
* `evaluate_trained_models.py`: A script for evaluating the f1 scores and accuracy of trained models. It also makes outputs predictions with the `-p` argument.
* `evaluate_topic_reasonability.py`: A script that returns the article text and its predicted label names. Used for evaluating whether the model's predictions seem reasonable to a human reader.
* `CRNN_training_log.json`: This file shows an excerpt of the output of the gridsearch we did for our CRNN model
* `competition/`: This directory contains the competition test set and our predictions

## Requirements
* Python 3.5+
* pip


## Installation
`pip install -r requirements.txt`


