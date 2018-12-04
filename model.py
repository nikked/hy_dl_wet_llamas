"""
The task is to learn to assign the correct labels to news articles.  The corpus
contains ~850K articles from Reuters.  The test set is about 10% of the articles.
The data is unextracted in XML files.

We're only giving you the code for downloading the data, and how to save the
final model. The rest you'll have to do yourselves.

Some comments and hints particular to the project:

- One document may belong to many classes in this problem, i.e., it's a multi-label
classification problem. In fact there are documents that don't belong to any class,
and you should also be able to handle these correctly. Pay careful attention to how
you design the outputs of the network (e.g., what activation to use) and what loss
function should be used.

- You may use word-embeddings to get better results. For example, you were already
using a smaller version of the GloVE  embeddings in exercise 4. Do note that these
embeddings take a lot of memory.

- In the exercises we used e.g., `torchvision.datasets.MNIST` to handle the loading
of the data in suitable batches. Here, you need to handle the dataloading yourself.
The easiest way is probably to create a custom `Dataset`. [See for example here for
a tutorial](https://github.com/utkuozbulak/pytorch-custom-dataset-examples).
"""
import json
import os
import sys
from pprint import pprint


REUTERS_JSON_FP = './train/reuters.json'


def main():

    if not os.path.exists(REUTERS_JSON_FP):
        print(f'Could not find file {REUTERS_JSON_FP}. Please run data_preprocess.py first.')
        sys.exit(1)

    with open(REUTERS_JSON_FP, 'r') as file:
        print(f'Opening file {REUTERS_JSON_FP}')
        reuters_articles = json.load(file)

    print(f'\nAmount of articles: {len(reuters_articles.keys())}\n')

    for key, content in reuters_articles.items():
        print(key)
        pprint(content['topic_codes'])
        pprint(content['headline'])
        pprint(content['text'])
        break


if __name__ == "__main__":

    main()
