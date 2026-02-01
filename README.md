# Fantasy Football LSTM Prediction Model

A machine learning project using Long Short-Term Memory (LSTM) neural networks to predict NFL player fantasy football performance based on historical game-by-game data.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Model Architecture](#model-architecture)
- [Fantasy Scoring System](#fantasy-scoring-system)
- [Data Format Requirements](#data-format-requirements)
- [Local Setup with NVIDIA CUDA](#local-setup-with-nvidia-cuda)
- [Running with Your Own Data](#running-with-your-own-data)
- [Understanding the Output](#understanding-the-output)
- [Notebooks Reference](#notebooks-reference)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project trains individual LSTM models for each NFL player to predict their fantasy football performance over a 16-game span. The model learns from a player's historical game-by-game fantasy point totals and forecasts future performance.

**Key Features:**
- Separate LSTM model trained per player
- Half-PPR fantasy scoring system
- Supports QB, RB, WR, and TE positions
- Predicts last 16 games for validation/testing

---

## Project Structure

```
fantasy-football/
├── clean_data.ipynb          # Data preprocessing (ACTIVE)
├── functions_model.ipynb     # LSTM model training and predictions (ACTIVE)
├── demo_model.ipynb          # Example walkthrough with Tom Brady (ACTIVE)
├── data_collection.ipynb     # Pro Football Reference scraper (NOT USED)
├── fantasy_data/             # Processed fantasy point CSVs
│   ├── qb_fantasy.csv
│   ├── rb_fantasy.csv
│   ├── wr_fantasy.csv
│   └── te_fantasy.csv
├── predictions/              # Model prediction outputs
│   ├── qb_prediction_last16.csv
│   ├── rb_prediction_last16.csv
│   ├── wr_prediction_last16.csv
│   └── te_prediction_last16.csv
├── actuals/                  # Actual values for comparison
│   ├── qb_actual_last16.csv
│   ├── rb_actual_last16.csv
│   ├── wr_actual_last16.csv
│   └── te_actual_last16.csv
├── requirements.txt          # Python dependencies
└── README.md
```

### Notebook Status

| Notebook | Status | Purpose |
|----------|--------|---------|
| `clean_data.ipynb` | **ACTIVE** | Processes raw player data into fantasy point CSVs |
| `functions_model.ipynb` | **ACTIVE** | Contains LSTM model and runs predictions |
| `demo_model.ipynb` | **ACTIVE** | Educational walkthrough using Tom Brady |
| `data_collection.ipynb` | **NOT USED** | Legacy scraper for Pro Football Reference (kept for reference only) |

---

## How It Works

### Workflow (2 Steps)

**Step 1: Data Cleaning (`clean_data.ipynb`)**

Reads raw player game log data and calculates Half-PPR fantasy points:

```
Input:  gridironai/player_data_YYYY.csv (years 2000-2020)
Output: fantasy_data/{position}_fantasy.csv
```

**Step 2: Model Training (`functions_model.ipynb`)**

Trains LSTM models and generates predictions:

```
Input:  fantasy_data/{position}_fantasy.csv
Output: predictions/{position}_prediction_last16.csv
        actuals/{position}_actual_last16.csv
```

### About data_collection.ipynb

This notebook contains a web scraper for Pro Football Reference but is **not part of the active workflow**. The project uses GridironAI data instead. It is kept in the repository for reference purposes only.

---

## Model Architecture

### LSTM Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Neurons** | 8 | Single LSTM layer with 8 units |
| **Epochs** | 400 | Training iterations per player |
| **Batch Size** | 1 | Single sample per batch (required for stateful LSTM) |
| **Stateful** | True | Maintains hidden state between batches |
| **Loss Function** | Mean Squared Error | Regression objective |
| **Optimizer** | Adam | Adaptive learning rate optimizer |
| **Scaler** | MinMaxScaler(-1, 1) | Normalizes data to [-1, 1] range |

### Network Architecture

```
Input (1 feature: previous game points)
    │
    ▼
┌─────────────────────────┐
│  LSTM Layer (8 units)   │
│  stateful=True          │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  Dense Layer (1 unit)   │
│  Linear activation      │
└─────────────────────────┘
    │
    ▼
Output (predicted points)
```

### Player Filtering Thresholds

| Position | Minimum Games Required |
|----------|------------------------|
| QB | 40+ games |
| RB, WR, TE | 30+ games |

Players with fewer games are excluded from training.

### Time Series Approach

The model uses **lag-1 supervised learning** transformation:

```
Original: [game1, game2, game3, game4, ...]
Transformed:
  X (input):  [0,     game1, game2, game3, ...]
  y (output): [game1, game2, game3, game4, ...]
```

This allows the model to learn: "Given the previous game's fantasy points, predict the next game's fantasy points."

### Training/Test Split

- **Training set:** All games except the last 16
- **Test set:** Last 16 games (approximately one full season)

---

## Fantasy Scoring System

### Half-PPR Formula

```python
HalfPPR = (
    (passing_yds / 25) +       # 1 point per 25 passing yards
    (passing_td * 4) +         # 4 points per passing TD
    (passing_int * -2) +       # -2 points per interception
    (rushing_yds / 10) +       # 1 point per 10 rushing yards
    (rushing_td * 6) +         # 6 points per rushing TD
    (receiving_rec * 0.5) +    # 0.5 points per reception (Half-PPR)
    (receiving_yds / 10) +     # 1 point per 10 receiving yards
    (receiving_td * 6) +       # 6 points per receiving TD
    (fumbles_lost * -2)        # -2 points per fumble lost
)
```

### Scoring Breakdown

| Stat | Points |
|------|--------|
| Passing Yard | 0.04 (1 per 25 yards) |
| Passing TD | 4 |
| Interception | -2 |
| Rushing Yard | 0.1 (1 per 10 yards) |
| Rushing TD | 6 |
| Reception | 0.5 |
| Receiving Yard | 0.1 (1 per 10 yards) |
| Receiving TD | 6 |
| Fumble Lost | -2 |

---

## Data Format Requirements

### Input Data Format (for `clean_data.ipynb`)

Your source data must be CSV files with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `player_id` | int | Unique player identifier |
| `franchise_id` | string | Team identifier |
| `position_id` | string | Position: QB, RB, WR, or TE |
| `name` | string | Player full name |
| `season` | int | Year (e.g., 2020) |
| `week` | int | Week number (1-17) |
| `playoffs` | int | 0 = regular season, 1 = playoffs |
| `passing_yds` | float | Passing yards |
| `passing_td` | float | Passing touchdowns |
| `passing_int` | float | Interceptions thrown |
| `rushing_yds` | float | Rushing yards |
| `rushing_td` | float | Rushing touchdowns |
| `receiving_rec` | float | Receptions |
| `receiving_yds` | float | Receiving yards |
| `receiving_td` | float | Receiving touchdowns |
| `fumbles_lost` | float | Fumbles lost |

**File naming convention:** `player_data_YYYY.csv` (e.g., `player_data_2023.csv`)

### Processed Fantasy Data Format

The output CSVs from `clean_data.ipynb` have this structure:

```
(row index = player name)
Columns: YYYY_gameN (e.g., 2000_game1, 2000_game2, ..., 2020_game17)
Values: Half-PPR fantasy points for that game (NaN if player did not play)
```

**Example:**
```csv
,2020_game1,2020_game2,2020_game3,...
Tom Brady,20.46,8.68,23.88,...
Aaron Rodgers,21.90,24.94,16.04,...
```

---

## Local Setup with NVIDIA CUDA

### Prerequisites

- NVIDIA GPU with CUDA support
- Python 3.8+
- CUDA Toolkit 11.x or 12.x
- cuDNN compatible with your CUDA version

### Step 1: Verify NVIDIA GPU

```bash
nvidia-smi
```

You should see your GPU listed with driver information.

### Step 2: Install CUDA Toolkit

Download from: https://developer.nvidia.com/cuda-downloads

Verify installation:
```bash
nvcc --version
```

### Step 3: Install cuDNN

Download from: https://developer.nvidia.com/cudnn (requires NVIDIA account)

Extract and copy files to your CUDA installation directory.

### Step 4: Create Python Environment

```bash
# Create virtual environment
python -m venv fantasy-lstm-env

# Activate (Windows)
fantasy-lstm-env\Scripts\activate

# Activate (Linux/Mac)
source fantasy-lstm-env/bin/activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn

# For CUDA-enabled TensorFlow
pip install tensorflow[and-cuda]
```

### Step 6: Verify GPU Detection

```python
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("Built with CUDA:", tf.test.is_built_with_cuda())
```

### GPU Memory Management (Recommended)

Add this at the start of notebooks to prevent TensorFlow from allocating all GPU memory:

```python
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

---

## Running with Your Own Data

### Option A: Using Pre-Processed Fantasy Data

If you already have fantasy point data in the correct format:

1. Place your CSVs in `fantasy_data/` with names:
   - `qb_fantasy.csv`
   - `rb_fantasy.csv`
   - `wr_fantasy.csv`
   - `te_fantasy.csv`

2. Ensure format matches:
   ```csv
   ,2023_game1,2023_game2,...,2023_game17
   Player Name,12.5,18.3,...,22.1
   ```

3. Run `functions_model.ipynb`

### Option B: Using Raw Game Log Data

If you have raw player statistics:

1. **Prepare your data directory:**
   Create a folder (e.g., `my_data/`) with files named `player_data_YYYY.csv`

2. **Modify `clean_data.ipynb`:**

   Update the file path in cell reading the data:
   ```python
   # Change from:
   df = pd.read_csv('gridironai/player_data_2020.csv')

   # To your path:
   df = pd.read_csv('my_data/player_data_2020.csv')
   ```

   Update the year range in the loop:
   ```python
   # Change from:
   for year in range(2000, 2021):

   # To your range:
   for year in range(2018, 2024):
   ```

3. **Run `clean_data.ipynb`** to generate fantasy point CSVs

4. **Run `functions_model.ipynb`** to train models and generate predictions

### Option C: Single Position Quick Test

To test with just one position, modify `functions_model.ipynb`:

```python
# Comment out other positions
# run_model(qb, 'qb')
run_model(rb, 'rb')  # Only run RBs
# run_model(wr, 'wr')
# run_model(te, 'te')
```

---

## Understanding the Output

### Prediction Files

Located in `predictions/` folder:

```csv
name,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
Aaron Rodgers,23.42,23.36,22.26,20.95,19.74,18.68,17.79,17.04,16.40,15.85,15.35,14.89,14.46,14.05,13.66,13.28
```

- **Columns 1-16:** Predicted fantasy points for each of the last 16 games
- **Values:** Half-PPR fantasy points

### Actual Files

Located in `actuals/` folder (same format):

```csv
name,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
Aaron Rodgers,30.76,18.7,24.52,29.58,3.8,27.32,22.54,28.9,25.4,21.74,25.64,23.5,30.9,18.32,25.14,26.0
```

### Comparing Results

```python
import pandas as pd

preds = pd.read_csv('predictions/qb_prediction_last16.csv')
actuals = pd.read_csv('actuals/qb_actual_last16.csv')

# Calculate total predicted vs actual for a player
player = 'Aaron Rodgers'
pred_total = preds[preds['name'] == player].iloc[0, 1:].sum()
actual_total = actuals[actuals['name'] == player].iloc[0, 1:].sum()

print(f"{player}: Predicted {pred_total:.1f}, Actual {actual_total:.1f}")
```

---

## Notebooks Reference

### demo_model.ipynb

**Purpose:** Educational walkthrough demonstrating the entire ML pipeline with a single player (Tom Brady).

**Key Sections:**
1. Data loading and player selection
2. Time series transformation with `series_to_supervised()`
3. Train/test split visualization
4. MinMaxScaler application
5. LSTM model training (400 epochs)
6. Prediction and inverse scaling
7. Results visualization with seaborn

**Use this notebook to:**
- Understand how the model works
- Experiment with individual players
- Debug issues with your data

### functions_model.ipynb

**Purpose:** Production notebook that trains models for all qualifying players.

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `series_to_supervised(data, lag=1)` | Transforms time series into supervised learning format |
| `ttsplit(dataframe)` | Splits data: training = all except last 16, test = last 16 games |
| `scale(train, test)` | Applies MinMaxScaler(-1, 1) fitted on training data |
| `fit_lstm(train, batch_size, num_epoch, neurons)` | Builds and trains stateful LSTM model |
| `forecast_lstm(model, batch_size, X)` | Makes single prediction |
| `invert_scale(scaler, X, value)` | Reverses scaling transformation |
| `run_model(dataframe, position)` | Main function: filters players, trains models, saves CSVs |

### clean_data.ipynb

**Purpose:** Preprocesses raw game log data into fantasy point matrices.

**Key Operations:**
1. Reads source CSV files (originally GridironAI, 2000-2020)
2. Filters out playoff games (`playoffs == 0`)
3. Calculates Half-PPR fantasy points
4. Pivots data: rows = players, columns = YYYY_gameN
5. Outputs position-specific CSVs

---

## Troubleshooting

### "No GPU detected"

1. Verify NVIDIA drivers: `nvidia-smi`
2. Check CUDA installation: `nvcc --version`
3. Ensure TensorFlow GPU version: `pip show tensorflow`
4. Test in Python:
   ```python
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   ```

### "Out of Memory" errors

1. Enable memory growth:
   ```python
   gpus = tf.config.list_physical_devices('GPU')
   for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu, True)
   ```

2. Reduce epochs (try 200 instead of 400)

3. Process one position at a time

### "Empty predictions file"

- Check that players meet minimum game thresholds (40 for QB, 30 for others)
- Verify fantasy data CSVs have correct format
- Ensure player names have no special characters

### "NaN values in predictions"

- Check for NaN in input data
- Verify scaler is fitted on training data only
- Ensure test set has valid values

### Model predictions are flat/constant

This is expected behavior - the LSTM tends toward mean reversion for players with inconsistent performance. To potentially improve:
- Increase neurons (try 16 or 32)
- Adjust epochs
- Add more lag features (modify `series_to_supervised()`)

---

## Author

Alvee Hoque

## Acknowledgments

- Data source: GridironAI
- LSTM approach inspired by time series forecasting literature
