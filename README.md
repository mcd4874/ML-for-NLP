# Problem Set 3

### Team Members
- William Duong (mcd4874)
- Mars Ballantyne (mxb9328)
- Christian Franco (caf1751)
- Steven Mirabito (stm4445)

## Installation

Assuming Anaconda is installed in `~/anaconda3`:
```
~/anaconda3/bin/pip install -r requirements.txt
```

## Usage

Tested with Anaconda Python 3.7.1.

Supported features: `genre`, `polarity`, `issue`

```
$ ~/anaconda3/bin/python train.py -h
usage: train.py [-h] FEATURE TRAINING_DATA OUTPUT_FILE

Trains a model to identify a feature of sentences.

positional arguments:
  FEATURE        the feature for which to train
  TRAINING_DATA  the path to the training corpra
  OUTPUT_FILE    where to save the trained model
  TEXT_MODEL_FILE where to save the trained text model

optional arguments:
  -h, --help     show this help message and exit
```

```
$ ~/anaconda3/bin/python test.py -h
usage: test.py [-h] FEATURE TRAINING_DATA MODEL_FILE

Uses a trained model to identify a feature of sentences.

positional arguments:
  FEATURE        the feature to identify
  TRAINING_DATA  the path to the test corpra
  MODEL_FILE     the path to the model to load
  TEXT_MODEL_FILE  where to load the trained text mode
  report_file  generate report file
  --report  Print a classification report

optional arguments:
  -h, --help     show this help message and exit
  --report, -r   Print a classification report
```

## Example

```
$ ~/anaconda3/bin/python train.py genre PS3_training_data.txt genre_model.pkl
Model saved to: genre_model.pkl

$ ~/anaconda3/bin/python test.py genre PS3_training_data.txt genre_model.pkl
0	This is definitely a must have if your state does not allow cell phone usage while driving.	POSITIVE	NONE	GENRE_B
[...]
```

We have included pre-trained models for each feature in the submission archive.

Use this 3 available command to test three case right away without re-trained.

For task 1
$ ~/anaconda3/bin/python test.py genre PS3_test_data_indexed.txt genre_model.pkl genre_cv_model.pkl genre_report.csv --r

For task 2
$ ~/anaconda3/bin/python test.py polarity PS3_test_data_indexed.txt polarity_model.pkl polarity_cv_model.pkl polarity_report.csv --r

For task 3
$ ~/anaconda3/bin/python test.py issue PS3_test_data_indexed.txt issue_model.pkl issue_cv_model.pkl issue_report.csv --r