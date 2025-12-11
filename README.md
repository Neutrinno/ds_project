ds_proj
==============================

Just project to do

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
## Installation and Environment Setup
------------

### Option 1 — Manual setup (via `venv`)

1. **Clone the repository**
    ```bash
    git clone https://github.com/Neutrinno/ds_project.git
    cd ds_project
    ```

2. **Create and activate virtual environment**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate   # for Linux/Mac
    .venv\Scripts\activate      # for Windows
    ```

3. **Install dependencies and project**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install -e .
    ```

4. **Install pre-commit hooks**
    ```bash
    pre-commit install
    ```

---

### Option 2 — Using `Makefile`

```bash
make setup       # creates .venv, installs dependencies and pre-commit
source .venv/bin/activate
```

---

### Option 3 — Using setup script

```bash
bash scripts/setup_environment.sh
source .venv/bin/activate
```

This script will automatically start Docker containers, create virtual environment, install dependencies and setup pre-commit hooks.

---

## S3 and MLFlow Setup

1. **Start MinIO and MLFlow containers**
   ```bash
   docker-compose up -d
   ```

2. **Access services:**
   - MinIO Console: http://localhost:9001 (login: minioadmin/minioadmin)
   - MinIO API: http://localhost:9000
   - MLFlow UI: http://localhost:5000

3. **Upload raw dataset to S3:**
   - Open MinIO Console: http://localhost:9001
   - Create bucket `iris-datasets` (if it doesn't exist)
   - Upload file `iris.zip` to the bucket with key `iris.zip`

4. **Run data processing pipeline**
   ```bash
   python -m src.data.make_dataset data/raw/iris.zip data/processed/iris_processed.csv
   ```

---

## Model Training

### Single Experiment

Train a model with specific hyperparameters using config file:

```bash
python -m src.models.train_model configs/hyperparameters.yaml data/processed/iris_processed.csv
```

Or use Makefile:

```bash
make train
```

### Grid Search Experiments

Run multiple experiments with hyperparameter grid search:

```bash
python scripts/run_experiments.py configs/hyperparameters_grid.yaml data/processed/iris_processed.csv
```

Or use Makefile:

```bash
make experiments
```

This will run 24 experiments with all combinations of:
- n_estimators: [50, 100, 200]
- max_depth: [3, 5, 7, 10]
- max_features: ["sqrt", "log2"]

### Grid Search in Docker Containers

Run each experiment in a separate Docker container:

```bash
python scripts/run_experiments.py configs/hyperparameters_grid.yaml data/processed/iris_processed.csv --docker
```

Or use Makefile:

```bash
make experiments-docker
```

### View Results

- MLFlow UI: http://localhost:5000
- Models are saved in S3 under `{experiment_name}/model.pkl`

---

## Training in Docker Container

Build and run training in a Docker container:

```bash
# Build training image
docker build -f Dockerfile.train -t ds-train .

# Run training
docker-compose --profile train up --build train
```

Or use Makefile:

```bash
make train-docker
```

---

## Configuration Files

### Experiment Configuration

Hyperparameters are defined in YAML config files located in `configs/` directory.

**Single experiment** (`configs/hyperparameters.yaml`):
```yaml
experiment_name: "iris_random_forest_v1"
model_type: "RandomForestClassifier"
hyperparameters:
  n_estimators: 100
  max_depth: 10
  max_features: "sqrt"
```

**Grid search** (`configs/hyperparameters_grid.yaml`):
```yaml
experiment_name: "iris_random_forest_grid_search"
model_type: "RandomForestClassifier"
hyperparameters_grid:
  n_estimators: [50, 100, 200]
  max_depth: [3, 5, 7, 10]
  max_features: ["sqrt", "log2"]
```

---

## Additional Files

```
├── configs/                    # Experiment configurations
│   ├── hyperparameters.yaml   # Single experiment config
│   └── hyperparameters_grid.yaml  # Grid search config
├── scripts/
│   ├── setup_environment.sh   # Environment setup script
│   └── run_experiments.py     # Grid search script
├── docker-compose.yml         # MinIO + MLFlow containers
└── Dockerfile.train           # Training container
```

---

## Development

### Linting and Type Checking

```bash
make lint                      # Run flake8
pre-commit run --all-files     # Run all pre-commit hooks (flake8 + mypy)
```

### Docker Commands

```bash
docker-compose up -d           # Start services
docker-compose down            # Stop services
docker-compose ps              # Check status
docker logs minio              # View MinIO logs
docker logs mlflow             # View MLFlow logs
```

---

## Quick Start

```bash
# 1. Setup environment
bash scripts/setup_environment.sh
source .venv/bin/activate

# 2. Verify containers are running
docker-compose ps

# 3. Train single model
make train

# 4. Run grid search
make experiments

# 5. View results in MLFlow UI
# Open http://localhost:5000 in browser
```
