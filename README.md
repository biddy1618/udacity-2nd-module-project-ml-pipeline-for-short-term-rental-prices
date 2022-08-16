# Build an ML Pipeline for Short-Term Rental Prices in NYC

## Project structure

```bash
📂 /path/to/project
┣━━ 📂 components                           # components folder
┃   ┣━━ 📂 get_data                         # `download_data` component
┃   ┃   ┣━━ 📂 data                         # details on setting up of ES service
┃   ┃   ┃   ┣━━ 📄 sample1.csv              # data source `sample1.csv`
┃   ┃   ┃   ┗━━ 📄 sample1.csv              # data source `sample2.csv`
┃   ┃   ┣━━ ❗ conda.yml                    # conda environment configuration
┃   ┃   ┣━━ ❗ MLproject                    # MLproject file configuration
┃   ┃   ┗━━ 🐍 run.py                       # main script for MLproject component
┃   ┣━━ 📂 test_regression_model            # `test_model` component
┃   ┃   ┣━━ ❗ conda.yml                    # conda environment configuration
┃   ┃   ┣━━ ❗ MLproject                    # MLproject file configuration
┃   ┃   ┗━━ 🐍 run.py                       # main script for MLproject component
┃   ┣━━ 📂 train_val_test_split             # `train_val_test_split` component
┃   ┃   ┣━━ ❗ conda.yml                    # conda environment configuration
┃   ┃   ┣━━ ❗ MLproject                    # MLproject file configuration
┃   ┃   ┗━━ 🐍 run.py                       # main script for MLproject component
┃   ┗━━ 📂 wandb_utils                      # helper library
┃       ┣━━ 🐍 __init__.py                  # conda environment configuration
┃       ┣━━ 🐍 log_artifact.py              # module for logging artifact
┃       ┗━━ 🐍 sanitize_path.py             # module for sanitizing absolute path
┣━━ 📂 cookie-mlflow-step                   # template for cookie cutter package
┃   ┗━━ ...                                 # ...
┣━━ 📂 images                               # static images/gifs
┃   ┗━━ ...                                 # ...
┣━━ 📂 src                                  # custom components
┃   ┣━━ 📂 basic_cleaning                   # `basic_cleaning` component
┃   ┃   ┣━━ ❗ conda.yml                    # conda environment configuration
┃   ┃   ┣━━ ❗ MLproject                    # MLproject file configuration
┃   ┃   ┗━━ 🐍 run.py                       # main script for MLproject component
┃   ┣━━ 📂 data_check                       # `data_check` component
┃   ┃   ┣━━ ❗ conda.yml                    # conda environment configuration
┃   ┃   ┣━━ ❗ MLproject                    # MLproject file configuration
┃   ┃   ┣━━ 🐍 conftest.py                  # PyTest parser and fixture definitions
┃   ┃   ┗━━ 🐍 test_data.py                 # PyTest tests
┃   ┣━━ 📂 eda                              # `eda` component
┃   ┃   ┣━━ ❗ conda.yml                    # conda environment configuration
┃   ┃   ┣━━ ❗ MLproject                    # MLproject file configuration
┃   ┃   ┗━━ 🐍 EDA.ipynb                    # notebook for EDA
┃   ┗━━ 📂 train_random_forest              # `train_random_forest` component
┃       ┣━━ ❗ conda.yml                    # conda environment configuration
┃       ┣━━ ❗ MLproject                    # MLproject file configuration
┃       ┣━━ 🐍 feature_engineering.py       # module for additional feature engineering functions and classes
┃       ┗━━ 🐍 run.py                       # main script for MLproject component
┣━━ 📄 .gitignore                           # gitignore file
┣━━ 📄 LICENSE.txt                          # license
┣━━ ❗ conda.yml                            # conda environment for the main componentn
┣━━ ❗ config.yaml                          # hydra configuration file
┣━━ ❗ environment.yml                      # conda environment for setting up the task (more in README-guide.md)
┣━━ ❗ MLproject                            # environment variables for docker compose
┣━━ 🐍 main.py                              # endpoints file
┣━━ 📄 README-guide.md                      # readme file on the task
┗━━ 📄 README.me                            # readme file on the project
```

## Commands

### Requirements

Create conda environment (version - __4.13.0__) and install mlflow package:
```bash
conda install -c conda-forge mlflow=1.14.1
``` 

### EDA

Command to run EDA:
```bash
mlflow run src/eda
```

### Modeling

Command used to fine-tune the model:
```bash
mlflow run . \
    -P steps=train_random_forest \
    -P hydra_options="modeling.max_tfidf_features=10,15,30 modeling.random_forest.max_features=0.1,0.33,0.5,0.75,1 -m"
```

Command to run the pipeline on the new sample:
```bash
mlflow run https://github.com/biddy1618/udacity-2nd-module-project-ml-pipeline-for-short-term-rental-prices.git \
    -v v1.0.3 -P hydra_options="download_data.sample='sample2.csv' main.experiment_name='production'"
```

### Note

- Component `data_check` requires reference artifact (data artifact with tag `reference`).
- Component `test_regression_model` requires production model (model artifact with tag `prod`).

## Changes made

- Implemented custom transformer class [`MeanTargetEncoder`](src/train_random_forest/feature_engineering.py) to deal with high-cardinality feature `neighbourhood`
- Refactored `config.yaml` file to hold all variable inputs (along with all components' `MLproject` and `run.py` files)

## Links

- [WANDB project link](https://wandb.ai/biddyasdiddy/udacity-mldevops-2nd-project-final)
- [Github link](https://github.com/biddy1618/udacity-2nd-module-project-ml-pipeline-for-short-term-rental-prices)