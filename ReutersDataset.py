"""

Basic usage:
python ReutersDataset.py -f reuters_sample.json

Depth limited usage (take first 1000 instances):
python ReutersDataset.py -f train/reuters.json -d 1000

"""
import sys
import json
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import argparse
from pprint import pprint


def load_data_example(reuters_dataset):

    dataloader = DataLoader(
        dataset=reuters_dataset,
        batch_size=3,
        shuffle=False
    )

    for explanatory_vars, labels in dataloader:
        pprint(len(labels))
        pprint(labels)
        pprint(explanatory_vars['headline'])
        # pprint(explanatory_vars['text'])
        print()


class ReutersDataset(Dataset):

    def __init__(self, json_path, max_depth):
        """
        function is where the initial logic happens like reading a csv, assigning transforms etc.
        """

        with open(json_path, 'r') as file:
            reuters_articles = json.load(file)

        depth_count = 0

        topic_codes_list = []
        headline_list = []
        text_list = []

        for article_id, content in reuters_articles.items():
            raw_text = ''
            for paragraph in content['text'].values():
                for line in paragraph:
                    raw_text += f'{line} '

            topic_codes_list.append(content['topic_codes'])
            headline_list.append(content['headline'])
            text_list.append(raw_text)

            depth_count += 1
            if depth_count >= max_depth:
                break

        self.data_len = len(topic_codes_list)

        self.label_arr = np.asarray(topic_codes_list)
        self.headline_arr = np.asarray(headline_list)
        self.text_arr = np.asarray(text_list)

    def __getitem__(self, index):
        """
        function returns the data and labels. __getitem__() needs to return a specific type for
        a single data point (like a tensor, numpy array etc.)
        """

        explanatory_vars = {}

        explanatory_vars['headline'] = self.headline_arr[index]
        explanatory_vars['text'] = self.text_arr[index]
        topic_codes = self.label_arr[index]

        return (explanatory_vars, topic_codes)

    def __len__(self):
        """
        returns count of samples you have
        """
        return self.data_len


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser()

    PARSER.add_argument('-f', '--filepath', type=str,
                        help='File path of source Reuters JSON file')
    PARSER.add_argument('-d', '--maxdepth', type=int,
                        help='Max no. of instances in the dataset')

    ARGS = PARSER.parse_args()

    MAX_DEPTH = 1e12
    if ARGS.maxdepth:
        MAX_DEPTH = ARGS.maxdepth

    JSON_FP = None
    if ARGS.filepath:
        JSON_FP = ARGS.filepath

    if not JSON_FP:
        print('Please give JSON filepath as argument')
        sys.exit(1)

    reuters_dataset = ReutersDataset(JSON_FP, MAX_DEPTH)

    load_data_example(reuters_dataset)
