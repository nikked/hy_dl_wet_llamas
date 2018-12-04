# News topic classification from Reuters articles


## File descriptions
* `data_preprocess.py`: Converts the collection of Reuters XMLs to a single large JSON file.
* `reuters_sample.json`: A small example snippet of the large JSON file.
* `model.ipynb` : Sample notebook for loading `reuters.json` as a Pandas DataFrame.
* `model.py`: This is a skeleton for the actual deep learning classificator. Currently it just opens the `reuters.json` and makes a Pandas DataFrame with relevant fields.
* `topic_codes.txt`: Descriptions of the target classes.
* `text_project.ipynb`: The exercise notebook as provided by instructors.


## Requirements
* Python 3.5+
* pip


## Installation
`pip install -r requirements.txt`


## Usage


### data_preprocess.py
`data_preprocess` fetches the `reuters.zip` from cs.helsinki server and makes a huge JSON dump from xml-files found from `reuters.zip`. The dump is by default is in the `train/` directory. Please see what the JSON file output look likes by opening `reuters_sample.json`

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
Currently just converts the `reuters.json` file to a Pandas DataFrame

### Data mismatch
There should be about 850K articles, but only 299773 xml-files where found. Either there is a bug or the issue is in source?
