import zipfile
import pandas as pd
import os
import xml.etree.ElementTree as ET


train_path = 'train/'

dl_file = 'reuters.zip'
dl_url = 'https://www.cs.helsinki.fi/u/jgpyykko/'
zip_path = os.path.join(train_path, dl_file)
if not os.path.isfile(zip_path):
    download_url(dl_url + dl_file, root=train_path, filename=dl_file, md5=None)

with zipfile.ZipFile(zip_path) as zip_f:
    zip_f.extractall(train_path)


def getHeadline(root):
    return root.find('headline').text


def getTitle(root):
    return root.find('title').text


def getCodes(root):
    metaElem = root.find('metadata')
    codesElem = metaElem.findall('codes')
    codes = []
    for c in codesElem:
        # Get only topic codes
        if c.attrib['class'] == 'bip:topics:1.0':
            for code in c:
                codes.append(code.attrib['code'])
    return codes


def getText(root):
    ps = root.find('text').findall('p')
    text = []
    for p in ps:
        text.append(p.text)
    return '\n'.join(text)


def parseXML(file):
    root = ET.parse(file).getroot()
    return getHeadline(root), getTitle(root), getText(root), getCodes(root)


def parseZip(file):
    zf = zipfile.ZipFile(file, 'r')
    for xml in zf.namelist():
        h, t, txt, cs = parseXML(zf.open(xml))
        headlines.append(h)
        titles.append(t)
        texts.append(txt)
        codes.append(cs)


data_path = 'train/REUTERS_CORPUS_2/'
headlines, titles, texts, codes = [], [], [], []
print('Processing data in', data_path)
for f in os.listdir(data_path):
    if f.startswith("1997") and f.endswith(".zip"):
        print('.', end='')
        parseZip(data_path + f)

df = pd.DataFrame({'headline': headlines, 'title': titles,
                   'text': texts, 'codes': codes})
print('\nCompressing dataframe to train/train.json.xz')
df.to_json('train/train.json.xz', orient='records', compression='xz')
