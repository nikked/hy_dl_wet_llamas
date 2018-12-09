"""
The above command downloads and extracts the data files into the train subdirectory.

The files can be found in train/, and are named as 19970405.zip, etc. You will have
to manage the content of these zips to get the data. There is a readme which has
links to further descriptions on the data.

The class labels, or topics, can be found in the readme file called train/codes.zip.
The zip contains a file called "topic_codes.txt". This file contains the special
codes for the topics (about 130 of them), and the explanation - what each code means.

The XML document files contain the article's headline, the main body text, and the
list of topic labels assigned to each article. You will have to extract the topics
of each article from the XML. For example: <code code="C18"> refers to the topic
"OWNERSHIP CHANGES" (like a corporate buyout).

You should pre-process the XML to extract the words from the article: the <headline>
element and the <text>. You should not need any other parts of the article.

"""


import os
import torch
from torchvision.datasets.utils import download_url
import zipfile
from pprint import pprint
import time
import xmltodict
import json

TRAIN_PATH = "train/"
UNZIP_FILE_PATH = os.path.join(TRAIN_PATH, "REUTERS_CORPUS_2")
UNZIPPED_ARTICLES = os.path.join(TRAIN_PATH, "reuters_data")
JSON_OUTPUT_PATH = os.path.join(TRAIN_PATH, "reuters.json")


def main():

    start_time = time.time()

    _fetch_data()

    _unzip_inner_reuters_files()

    _make_json_of_xml_files()

    end_time = time.time()

    print("Time passed: {} seconds".format(round(end_time - start_time, 2)))


def _fetch_data():
    train_path = "train/"

    dl_file = "reuters.zip"
    dl_url = "https://www.cs.helsinki.fi/u/jgpyykko/"
    zip_path = os.path.join(train_path, dl_file)

    print(f"\nFetching file to path: {zip_path}")
    if not os.path.isfile(zip_path):
        download_url(dl_url + dl_file, root=train_path, filename=dl_file, md5=None)
        print("Download successful")
    else:
        print(f"File already downloaded. Continuing.")

    print(f"\nUnzipping file to path: {UNZIP_FILE_PATH}")
    if not os.path.isdir(UNZIP_FILE_PATH):
        with zipfile.ZipFile(zip_path) as zip_f:
            zip_f.extractall(train_path)
        print("Unzip successful")
    else:
        print(f"File already unzipped to path. Continuing.")


def _unzip_inner_reuters_files():

    print("\nProceeding to unzip inner reuters files")

    if not os.path.exists(UNZIPPED_ARTICLES):
        os.makedirs(UNZIPPED_ARTICLES)

    for reuters_zip_filename in os.listdir(UNZIP_FILE_PATH):
        reuters_zip_filepath = os.path.join(UNZIP_FILE_PATH, reuters_zip_filename)

        if reuters_zip_filepath.endswith(".zip"):

            unzipped_reuters_filepath = os.path.join(
                UNZIPPED_ARTICLES, reuters_zip_filename[:-4]
            )

            if not os.path.isdir(unzipped_reuters_filepath):
                with zipfile.ZipFile(reuters_zip_filepath) as zip_f:
                    zip_f.extractall(unzipped_reuters_filepath)


def _make_json_of_xml_files():

    mega_dict = {}
    for idx, xml_dir in enumerate(os.listdir(UNZIPPED_ARTICLES)):

        if idx % 3 == 0:
            print(f"Handling file {idx+1}/{len(os.listdir(UNZIPPED_ARTICLES))}")
        xml_dir_fp = os.path.join(UNZIPPED_ARTICLES, xml_dir)

        if xml_dir.startswith("1997"):
            for xml_doc_fn in os.listdir(xml_dir_fp):
                xml_doc_fp = os.path.join(xml_dir_fp, xml_doc_fn)

                with open(xml_doc_fp, "r", encoding="ISO-8859-1") as file:
                    xml_as_str = file.read()

                xml_as_dict = xmltodict.parse(xml_as_str)

                newsitem = xml_as_dict["newsitem"]

                headline = newsitem["headline"]
                text = newsitem["text"]

                code_groups = newsitem["metadata"]["codes"]
                topic_codes = []
                for code_group in code_groups:

                    if not isinstance(code_group, str):
                        if "topics" in code_group["@class"]:
                            for topic_code in code_group["code"]:
                                if topic_code == "@code":
                                    topic_codes.append(code_group["code"]["@code"])
                                    break
                                else:
                                    topic_codes.append(topic_code["@code"])
                headline = newsitem["headline"]
                text = newsitem["text"]

                dict_key = f"{xml_dir}_{xml_doc_fn[:-4]}"
                mega_dict[dict_key] = {
                    "topic_codes": topic_codes,
                    "headline": headline,
                    "text": text,
                }

    with open(JSON_OUTPUT_PATH, "w") as file:
        json.dump(mega_dict, file)


if __name__ == "__main__":
    main()
