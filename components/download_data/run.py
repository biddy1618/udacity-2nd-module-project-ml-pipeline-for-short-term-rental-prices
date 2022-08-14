#!/usr/bin/env python
'''
This script downloads a data source to a W&B
'''
import argparse
import logging
import os
import yaml

import wandb

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
    '''
    Main script to run for the MLproject component.

    Args:
        args: Arguments for MLproject component.
    '''
    run = wandb.init(job_type=JOB_TYPE)
    run.config.update(args)

    logger.info(f'Returning sample {args.sample}.')
    logger.info('Checking if file exists.')
    try:
        data_path = os.path.join(os.getcwd(), 'data', args.sample)
        if not os.path.isfile(os.path.join(os.getcwd(), 'data', args.sample)):
            raise FileNotFoundError
        logger.info(f'Found data source - {data_path}.')
    except FileNotFoundError as e:
        logger.error(f'Data source not found - {data_path}.')
        raise e
    
    logger.info(f"Uploading {args.artifact_name} to Weights & Biases")
    log_artifact(
        args.artifact_name,
        args.artifact_type,
        args.artifact_description,
        data_path,
        run,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download URL to a local destination")

    parser.add_argument("sample", type=str, help="Name of the sample to download")

    parser.add_argument("artifact_name", type=str, help="Name for the output artifact")

    parser.add_argument("artifact_type", type=str, help="Output artifact type.")

    parser.add_argument(
        "artifact_description", type=str, help="A brief description of this artifact"
    )

    args = parser.parse_args()

    go(args)
