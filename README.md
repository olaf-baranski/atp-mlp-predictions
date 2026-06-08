# ATP matches MLP Predictions

This project predicts ATP men's singles match winners using a simple MLP (feed-forward neural network) trained on historical tennis match data.

## Project structure (planned)
- `download_data.py` – downloads raw data (the full dataset is NOT included in the submission archive)
- `prepare_data.py` – builds features and creates train/val/test splits
- `train.py` – trains the MLP model and saves it to `models/`
- `evaluate.py` – evaluates the model and saves metrics to `reports/`
- `predict.py` – generates predictions for new matchups from a CSV input file

## Quickstart (to be finalized)
```bash
pip install -r requirements.txt
python download_data.py
python prepare_data.py
python train.py
python evaluate.py
python predict.py --input examples/input_matches.csv --output examples/output_predictions.csv
