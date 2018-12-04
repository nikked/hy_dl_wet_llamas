# News topic classification from Reuters articles


## Requirements
* Python 3.5+
* pip


## Installation
`pip install -r requirements.txt`


## Usage


### Data_preprocess.py

> Please see what the JSON file output look likes by opening `reuters_sample.json`


The only working component at the moment is `data_preprocess`. This script creates a super large `reuters.json` file which by default is in the `train/` directory.

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
