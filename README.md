# twitter-sentiment-analysis

Originally started to help get my head around attention mechanisms, this small project implements several versions of attention backed by more conventional RNNs.
Included is a demo that allows you to experiment with the attention mechanism by selectively dropping tokens out of the input. This allows you to see how the inclusion 
or exclusion of different words affect both the attention calculation and the end prediction.

## Requirements
  * Python>=3.6
  * NumPy
  * tqdm
  * TensorFlow==1.13
  * Spacy
  * Pandas
  * Flask (only for demo)
  * WordSegment

## Datasets

### Sentiment 140

Sentiment 140 is a dataset consisting of 1.6 million tweets extracted from the Twitter API and labelled with the polarity
of the tweet. The polarity was determined by assuming that any tweets with positive emoticons e.g. :) were positive and
any with negative emoticons e.g. :( were negative. This is one of the larger open-source sentiment analysis datasets and
we therefore saw fit to train on it.

To train on Sentiment 140 data run the command: ```python main.py --dataset sent_140```

### SemEval

SemEval is a NLP workshop aimed at pushing the state of the art in the field of semantic evaluation. Every year the release
a set of challenges and a high-quality dataset for each with the aim of producing novel approaches within the given area. Within this
project we use the twitter datset from the 2017 set of challenges.

To train on SemEval 2017 data run the command: ```python main.py --dataset semeval```

## Usage

1 - Install Requirements

For a full list of the requirements refer to requirements.txt. The script below installs everything you require.

```
pip install

# Download Spacy Model
python -m spacy download en_core_web_sm
```

2 - Download pre-trained embeddings

The pre-trained models are trained using 300D cased GloVe vectors, place these into the /data/embeddings folder. If
not, you can adjust the embeddings key in the /data/defaults.json to point to any embeddings you have already downloaded.

- [GloVe Vectors](https://nlp.stanford.edu/projects/glove/)

3 - Run preprocessing

Before we can train, we pre-process the raw data into a series of tf-record files. Within this step we add POS tags,
create word/character matrices derived from GLOVE and handle OOV words. To run preprocessing use the command below.

```
python main.py --mode preprocess --datset semeval
```

4 - Run training

To start training simply run the following command:

```
python main.py --mode train --run_name this_is_a_new_run_name
```

5 - Run Demo (Optional)

The demo can be started with ```python main.py --mode demo``` or ```python demo.py```.
This starts a flask server both serving the demo as well as running a small APIon localhost:5000. 

## Contact information

For help or issues using this repo, please submit a GitHub issue.
