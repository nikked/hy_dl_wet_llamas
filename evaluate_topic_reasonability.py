from pprint import pprint
import pandas as pd
import numpy as np

DF_COMPETITION_FILE_PATH = 'competition.json.xz'


def load_training_set_as_df(df_filepath):
    df = pd.read_json(df_filepath, compression='xz')
    return df


def initialize_topic_codes():
    topic_codes = []
    topic_desc = []
    with open("./topic_codes.txt", "r") as f:
        tc = f.readlines()
        tc = tc[2:]
        for x in tc:
            c, l = x.split("\t", 1)
            topic_codes.append(c.strip())
            topic_desc.append(l.strip())

    return np.array(topic_codes), np.array(topic_desc)


topic_codes, topic_desc = initialize_topic_codes()

# pprint(topic_desc)
# pprint(topic_codes)


df = load_training_set_as_df(DF_COMPETITION_FILE_PATH)
# print(df.head())


results = np.loadtxt('competition_results.txt')

results_tf = results == 1

# print(results_tf)

argwheres = np.argwhere(results)


for row in range(len(results_tf)):

    pprint(df.text[row][:1000])
    pprint(topic_desc[results_tf[row]])
    print()

    if row > 10:
        break
