'''
Main components file
'''

import json

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

_steps = [
    'download_data',
    'basic_cleaning',
    'data_check',
    'data_split',
    'train_random_forest',
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to 'prod' before you can run this,
    # then you need to run this step explicitly
#    'test_regression_model'
]


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ['WANDB_PROJECT'] = config['main']['project_name']
    os.environ['WANDB_RUN_GROUP'] = config['main']['experiment_name']

    # Get the path at the root of the MLflow project
    root_path = hydra.utils.get_original_cwd()

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != 'all' else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if 'download_data' in active_steps:
            # Download file and load in W&B
            # _ = mlflow.run(
            #     f"{config['main']['components_repository']}/get_data",
            #     entry_point='main',
            #     version='main',
            #     parameters={
            #         'sample': config['download_data']['sample'],
            #         'artifact_name': config['download_data']['artifact_name'],
            #         'artifact_type': config['download_data']['artifact_type'],
            #         'artifact_description': config['download_data']['artifact_description']
            #     },
            # )
            _ = mlflow.run(
                os.path.join(root_path, 'components', 'download_data'),
                entry_point='main',
                parameters={
                    'sample': config['download_data']['sample'],
                    'artifact_name': config['download_data']['artifact_name'],
                    'artifact_type': config['download_data']['artifact_type'],
                    'artifact_description': config['download_data']['artifact_description']
                },
            )

        if 'basic_cleaning' in active_steps:
            ##################
            # Implement here #
            ##################
            _ = mlflow.run(
                os.path.join(root_path, 'src', 'basic_cleaning'),
                'main',
                parameters={
                    'artifact_input': f"{config['download_data']['artifact_name']}:latest",
                    'artifact_name': config['basic_cleaning']['artifact_name'],
                    'artifact_type': config['basic_cleaning']['artifact_type'],
                    'artifact_description': config['basic_cleaning']['artifact_description'],
                    'min_price': config['basic_cleaning']['etl']['min_price'],
                    'max_price': config['basic_cleaning']['etl']['max_price']
                }
            )

        if 'data_check' in active_steps:
            ##################
            # Implement here #
            ##################
            _ = mlflow.run(
                os.path.join(root_path, 'src', 'data_check'),
                'main',
                parameters={
                    'artifact_input': f"{config['basic_cleaning']['artifact_name']}:latest",
                    'artifact_reference': f"{config['basic_cleaning']['artifact_name']}:reference",
                    'kl_threshold': config['data_check']['kl_threshold'],
                    'min_price': config['basic_cleaning']['etl']['min_price'],
                    'max_price': config['basic_cleaning']['etl']['max_price']
                }
            )

        if 'data_split' in active_steps:
            ##################
            # Implement here #
            ##################
            # _ = mlflow.run(
            #     f"{config['main']['components_repository']}/train_val_test_split",
            #     entry_point='main',
            #     version='main',
            #     parameters={
            #         'input': f"{config['components']['basic_cleaning']['artifact_name']}:latest",
            #         'test_size': config['modeling']['test_size'],
            #         'random_seed': config['modeling']['random_seed'],
            #         'stratify_by': config['modeling']['stratify_by']
            #     }
            # )
            _ = mlflow.run(
                os.path.join(root_path, 'components', 'train_val_test_split'),
                entry_point='main',
                parameters={
                    'artifact_input': f"{config['components']['basic_cleaning']['artifact_name']}:latest",
                    'test_size': config['modeling']['test_size'],
                    'random_seed': config['modeling']['random_seed'],
                    'stratify_by': config['modeling']['stratify_by']
                },
            )

        if 'train_random_forest' in active_steps:

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath('rf_config.json')
            with open(rf_config, 'w+') as fp:
                json.dump(dict(config['modeling']['random_forest'].items()), fp)  # DO NOT TOUCH

            # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
            # step

            ##################
            # Implement here #
            ##################
            _ = mlflow.run(
                os.path.join(root_path, 'src', 'train_random_forest'),
                'main',
                parameters={
                    'trainval_artifact': 'trainval_data.csv:latest',
                    'val_size': config['modeling']['val_size'],
                    'random_seed': config['modeling']['random_seed'],
                    'stratify_by': config['modeling']['stratify_by'],
                    'rf_config': rf_config,
                    'max_tfidf_features': config['modeling']['max_tfidf_features'],
                    'output_artifact': 'random_forest_export'
                }
            )

        if 'test_regression_model' in active_steps:

            ##################
            # Implement here #
            ##################
            # _ = mlflow.run(
            #     f"{config['main']['components_repository']}/test_regression_model",
            #     entry_point='main',
            #     version='main',
            #     parameters={
            #         'mlflow_model': 'random_forest_export:prod',
            #         'test_dataset': 'test_data.csv:latest'
            #     }
            # )
            _ = mlflow.run(
                os.path.join(root_path, 'components', 'test_regression_model'),
                entry_point='main',
                parameters={
                    'mlflow_model': 'random_forest_export:prod',
                    'test_dataset': 'test_data.csv:latest'
                },
            )


if __name__ == '__main__':
    go()
