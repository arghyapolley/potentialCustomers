# Potential Customers Predictor

This repository contains a simple Python pipeline that uses [LightGBM](https://lightgbm.readthedocs.io/en/latest/) to predict whether a customer will make a purchase based on their past interaction data. The model reads from a Parquet dataset and outputs a ranked CSV of customers by their likelihood to purchase.

## Features

- Reads customer session/behavioral data from `.parquet` files.
- Prepares and cleans numerical tracking features (e.g., lifetime spend, order values, product categories).
- Trains a `LightGBM` classification model to predict user conversion (`PurchasedQuantityAfterSample`).
- Ranks customers based on their purchase probability.
- Outputs the predictions to a CSV file (`ranked_customers_by_purchase_probability.csv`).

## Prerequisites

Ensure you have Python 3 installed. You can install all required dependencies by running:

```bash
pip install pandas lightgbm pyarrow fastparquet shap scikit-learn openpyxl
```

**Note for macOS Users:** `lightgbm` requires OpenMP. If you encounter an error related to OpenMP, install `libomp` via Homebrew:

```bash
brew install libomp
```

## Usage

You can run the script from the command line by providing the path to your parquet data file as an argument.

```bash
python main.py <path_to_parquet_file>
```

### Example

```bash
python main.py "./data/customer_data.parquet"
```

## Data Requirements

The script expects a `.parquet` file with tracking features like:
- `cust_orders_lifetime`
- `cust_spend_lifetime`
- `cust_avg_order_value`
- `spend_in_category`
- `PurchasedQuantityAfterSample` (Used as the target column)

*(Note: Data files are ignored via `.gitignore` and are not included in this repository to prevent leaking sensitive information.)*

## Output format
The script creates `ranked_customers_by_purchase_probability.csv`, containing the original dataset merged with a `purchase_probability` column and optionally a `customer_type` column (differentiating "Actual Buyer" vs. "Predicted Buyer" based on original dataset target).

## License

This project is intended for demonstration purposes. Use it freely to rank and identify potential customers!
