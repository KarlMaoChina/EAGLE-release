# EAGLE: Enhanced AI for Gallbladder Lesion Evaluation

This repository contains the official PyTorch implementation for **EAGLE (Enhanced AI for Gallbladder Lesion Evaluation)**, an artificial intelligence (AI)-based model for the early detection of Gallbladder Cancer (GBC) from non-contrast CT images. This work is based on the research paper: *Harnessing multimodal artificial intelligence for enhanced early diagnostics in gallbladder cancer*.

EAGLE is an integrated AI-based diagnostic platform that incorporates advanced organ segmentation and multimodal feature fusion. The diagnostic system is driven by a novel **Bidirectional Adaptive Modal Fusion (BiAMF)** framework, which integrates clinical parameters, radiomic features, and deep learning representations extracted from non-contrast CT scans.

## Project Structure

```
.
├── main.py                 # Main script to run training
├── configs/
│   └── config.yaml         # Configuration file for paths & hyperparameters
├── src/
│   ├── __init__.py
│   ├── data.py             # Dataset class (MultiModalDataset)
│   ├── models.py           # All model-related classes
│   ├── trainer.py          # The main training and evaluation logic
│   └── utils.py            # Helper functions
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\\Scripts\\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Configure Data Paths:**

    Before running the training, you must update the paths in the configuration file `configs/config.yaml` to point to your datasets and pre-trained models.

    ```yaml
    paths:
      csv_path: "/path/to/your/dataset.csv"
      data_path: "/path/to/your/image_data"
      cl_data_path: "/path/to/your/clinical_data.xlsx"
      ra_data_path: "/path/to/your/radiomics_features.csv"
      experiment_base_path: "/path/to/your/image_model_checkpoints"
      concat_base_path: "/path/to/your/concat_model_checkpoints"
      ra_feature_names_path: "/path/to/your/ra_feature_names.txt"
    ```

2.  **Run Training:**

    To start the training process, run the `main.py` script. You can optionally point to a different configuration file using the `--config` argument.

    ```bash
    python main.py --config configs/config.yaml
    ```

    The script will create a new experiment directory inside `results/` containing logs, model checkpoints, and other artifacts.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
