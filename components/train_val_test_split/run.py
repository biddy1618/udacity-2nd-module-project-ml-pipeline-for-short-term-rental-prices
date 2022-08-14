#!/usr/bin/env python
'''
This script splits the provided dataframe in test and remainder.
'''
import argparse
import logging
import os
import yaml


import wandb
from wandb_utils.log_artifact import log_artifact

import tempfile

import pandas as pd
from sklearn.model_selection import train_test_split

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

    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    logger.info(f'Fetching artifact {args.artifact_input}')
    artifact_local_path = run.use_artifact(args.artifact_input).file()

    df = pd.read_csv(artifact_local_path)

    logger.info('Splitting trainval and test')
    trainval, test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=df[args.stratify_by] if args.stratify_by != 'none' else None,
    )

    # Save training and validation data split as artifact
    logger.info(f'Uploading training and validation dataset - {args.artifact_trainval_name}')
    with tempfile.NamedTemporaryFile('w') as fp:

        trainval.to_csv(fp.name, index=False)

        log_artifact(
            args.artifact_trainval_name,
            args.artifact_trainval_type,
            args.artifact_trainval_description,
            fp.name,
            run,
        )
    
    # Save test data split as artifact
    logger.info(f'Uploading test dataset - {args.artifact_test_name}')
    with tempfile.NamedTemporaryFile('w') as fp:

        test.to_csv(fp.name, index=False)

        log_artifact(
            args.artifact_test_name,
            args.artifact_test_type,
            args.artifact_test_description,
            fp.name,
            run,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split test and remainder')

    parser.add_argument('artifact_input', type=str, help='Input artifact to split')

    parser.add_argument(
        '--artifact_trainval_name', type=str, help='Name of the training and validation artifact'
    )

    parser.add_argument(
        '--artifact_trainval_type', type=str, help='Type of the training and validation artifact'
    )

    parser.add_argument(
        '--artifact_trainval_description', type=str, help='Description of the training and validation artifact'
    )

    parser.add_argument(
        '--artifact_test_name', type=str, help='Name of the test artifact'
    )

    parser.add_argument(
        '--artifact_test_type', type=str, help='Type of the test artifact'
    )

    parser.add_argument(
        '--artifact_test_description', type=str, help='Description of the test artifact'
    )

    parser.add_argument(
        '--test_size', type=float, help='Size of the test split. Fraction of the dataset, or number of items'
    )

    parser.add_argument(
        '--random_seed', type=int, help='Seed for random number generator', default=42, required=False
    )

    parser.add_argument(
        '--stratify_by', type=str, help='Column to use for stratification', default='none', required=False
    )

    args = parser.parse_args()

    go(args)
