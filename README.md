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

