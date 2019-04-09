# Problem Set 3

### Team Members
- William Duong (mcd4874)
- Mars Ballantyne (mxb9328)
- Christian Franco (caf1751)
- Steven Mirabito (stm4445)

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

optional arguments:
  -h, --help     show this help message and exit
```

## Example

```
$ ~/anaconda3/bin/python train.py genre PS3_training_data.txt genre_model.bin
Model saved to: genre_model.bin

$ ~/anaconda3/bin/python test.py genre PS3_training_data.txt genre_model.bin
              precision    recall  f1-score   support

     GENRE_A       0.99      0.99      0.99      1280
     GENRE_B       0.99      0.99      0.99      1284

   micro avg       0.99      0.99      0.99      2564
   macro avg       0.99      0.99      0.99      2564
weighted avg       0.99      0.99      0.99      2564
```