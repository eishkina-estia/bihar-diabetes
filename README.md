# Diabetes

Example of separation of data loading and processing, model training and inference logic.

## Files

- `data/download_data.py`: Script for loading data into a local SQLite database (`data\diabetes.db`)
- `notebooks/EDA.ipynb`: Jupyter notebook for basic exploratory data analysis (EDA).
- `model/train.py`: Script for training a model and saving it to the `model-registry` folder
- `model/evaluate.py`: Script for evaluating the trained model using the test dataset.
- `common.py`: Script for loading and preparing config constants from `config.yml` file.
- `requirements.txt`: List of required Python packages.

## Run 

### 1. Load data into the database
Run the following command to download and store the dataset in an SQLite database:
```shell
$ python -m data.download_data
```
This creates the `data/diabetes.db` file, a [lightweight disk-based database](https://docs.python.org/3/library/sqlite3.html) containing 2 tables: train and test.

### 2. Train and save the model
To train the model and save it, run:
```shell
$ python -m model.train
```
This creates the `models/diabetes.model` file, which contains a serialized regression model.

### 3. Test inference
To test the trained model using the test dataset, run:
```shell
python -m model.evaluate
```
This script loads the previously saved model and evaluates its performance.

