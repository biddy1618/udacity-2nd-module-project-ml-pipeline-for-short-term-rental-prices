#!/usr/bin/env python
'''
This step takes the best model, tagged with the 'prod' tag, and tests it against the test dataset
'''
import argparse
import logging
import os
import yaml

import wandb
import mlflow

import pandas as pd
from sklearn.metrics import mean_absolute_error

from wandb_utils.log_artifact import log_artifact


logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s')
logger = logging.getLogger()

JOB_TYPE = 'default'
try:
    with open(os.path.join(os.getcwd(), 'MLproject')) as file:
        doc = yaml.full_load(file)
    JOB_TYPE = doc['name']
except FileNotFoundError as e:
    logger.error('`MLproject` file not found in `download_data` component.')
    raise e
except KeyError as e:
    logger.error('`MLproject` file doesn\'t have `name` as key in `download_data` component.')
    raise e

def go(args):

    run = wandb.init(job_type=JOB_TYPE)
    run.config.update(args)

    logger.info('Downloading artifacts')
    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    model_local_path = run.use_artifact(args.artifact_model_name).download()

    # Download test dataset
    test_dataset_path = run.use_artifact(args.artifact_test_name).file()

    # Read test dataset
    X_test = pd.read_csv(test_dataset_path)
    y_test = X_test.pop('price')

    logger.info('Loading model and performing inference on test set')
    sk_pipe = mlflow.sklearn.load_model(model_local_path)
    y_pred = sk_pipe.predict(X_test)

    logger.info('Scoring')
    r_squared = sk_pipe.score(X_test, y_test)

    mae = mean_absolute_error(y_test, y_pred)

    logger.info(f'Score: {r_squared}')
    logger.info(f'MAE: {mae}')

    # Log MAE and r2
    run.summary['r2'] = r_squared
    run.summary['mae'] = mae


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test the provided model against the test dataset')

    parser.add_argument(
        '--artifact_model_name',
        type=str, 
        help='Input MLFlow model',
        required=True
    )

    parser.add_argument(
        '--artifact_test_name',
        type=str, 
        help='Test dataset',
        required=True
    )

    args = parser.parse_args()

    go(args)
