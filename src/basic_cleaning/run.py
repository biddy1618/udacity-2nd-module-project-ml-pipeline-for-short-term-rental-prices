'''
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact.

Author: Dauren Baitursyn
'''
import argparse
import logging
import os
import yaml

import wandb

import pandas as pd


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

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info('Downloading artifact')
    local_path = wandb.use_artifact(args.artifact_input).file()

    logger.info('Reading artifact')
    df = pd.read_csv(local_path)

    # Drop outliers
    logger.info('Dropping outliers')
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    logger.info('Converting last_review to datetime')
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Drop rows in the dataset that are not in the proper geolocation
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    logger.info('Logging artifact')
    filename = 'clean_sample.csv' 
    df.to_csv(filename, index=False)
    artifact = wandb.Artifact(
        args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description
    )
    artifact.add_file(filename)
    run.log_artifact(artifact) 
    artifact.wait()

    os.remove(filename)
    logger.info('Finished running component')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='A very basic data cleaning')


    parser.add_argument(
        '--artifact_input', 
        type=str,
        help='Name of the input artifact to clean',
        required=True
    )

    parser.add_argument(
        '--artifact_name', 
        type=str,
        help='Name of the artifact to save after cleaning',
        required=True
    )

    parser.add_argument(
        '--artifact_type', 
        type=str,
        help='Type for the artifact to be saved',
        required=True
    )

    parser.add_argument(
        '--artifact_description', 
        type=str,
        help='Description of the artifact',
        required=True
    )

    parser.add_argument(
        '--min_price', 
        type=float,
        help='Minimum price to filter the data',
        required=True
    )

    parser.add_argument(
        '--max_price', 
        type=float,
        help='Maximum price to filter the data',
        required=True
    )


    args = parser.parse_args()

    go(args)
