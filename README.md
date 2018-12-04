# News topic classification from Reuters articles


## File descriptions
* `data_preprocess.py`: Only working file at the moment. Converts the collection of Reuters XMLs to a single large JSON file.
* `reuters_sample.json`: A small example snippet of the large JSON file
* `model.py`: This is a skeleton for the actual deep learning classificator. Currently it just opens the `reuters.json` and prints the first article
* `topic_codes.txt`: Descriptions of the target classes
* `text_project.ipynb`: The exercise notebook as provided by instructors


## Requirements
* Python 3.5+
* pip


## Installation
`pip install -r requirements.txt`


## Usage


### data_preprocess.py

> There seems to be a bug at this scripts since only 299773 are retrieved at the moment.


The only working component at the moment is `data_preprocess`. This script creates a super large `reuters.json` file which by default is in the `train/` directory. Please see what the JSON file output look likes by opening `reuters_sample.json`

More specifically the `data_preprocess.py`:
1. Fetches the Reuters articles from `https://www.cs.helsinki.fi/u/jgpyykko/`
2. Unzips the content to `REUTERS_CORPUS2`. This file contains individual zip files from APR to AUG 1997 of Reuters articles
3. `REUTERS_CORPUS2` day-zips are further unzipped to `reuters_data`
4. Finally a `reuters.json` file is created which contains in json format all the fields we are interested from
    * text
    * headline
    * topic codes

Run the file by:
`python data_preprocess.py`


### model.py

This file currently just holds an iterator for the `reuters.json`