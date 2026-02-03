# PyTorch Migration Tasks

## Overview
Migrate fantasy football LSTM model from TensorFlow/Keras to PyTorch for Windows GPU support.

## Current TensorFlow Components to Convert

### 1. Imports
**TensorFlow (current):**
```python
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
```

**PyTorch (new):**
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
```

---

### 2. LSTM Model Definition
**TensorFlow (current):**
```python
model = Sequential()
model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
```

**PyTorch (new):**
```python
class FantasyLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=8, output_size=1):
        super(FantasyLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        output = self.fc(lstm_out[:, -1, :])
        return output, hidden
```

---

### 3. Training Loop
**TensorFlow (current):**
```python
for i in range(num_epoch):
    model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
    model.reset_states()
```

**PyTorch (new):**
```python
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(num_epochs):
    model.train()
    hidden = None  # Reset states each epoch
    optimizer.zero_grad()
    output, hidden = model(X_tensor, hidden)
    loss = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()
```

---

### 4. Prediction Function
**TensorFlow (current):**
```python
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]
```

**PyTorch (new):**
```python
def forecast_lstm(model, X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).reshape(1, 1, -1)
        if torch.cuda.is_available():
            X_tensor = X_tensor.cuda()
        output, _ = model(X_tensor)
        return output.cpu().numpy()[0, 0]
```

---

## Files to Update

| File | Changes Needed |
|------|----------------|
| `demo_model.ipynb` | Update imports, model class, training loop, prediction |
| `functions_model.ipynb` | Update `fit_lstm()` and `forecast_lstm()` functions |

---

## Migration Checklist

- [ ] Install PyTorch with CUDA support
- [ ] Verify GPU detection with `torch.cuda.is_available()`
- [ ] Create `FantasyLSTM` class in `functions_model.ipynb`
- [ ] Update `fit_lstm()` function to use PyTorch training loop
- [ ] Update `forecast_lstm()` function for PyTorch inference
- [ ] Add device handling (CPU/GPU) throughout
- [ ] Update `demo_model.ipynb` imports and model usage
- [ ] Test with single player (Tom Brady) to validate
- [ ] Run full model on all positions
- [ ] Compare results with previous TensorFlow outputs

---

## Notes

- **sklearn stays the same**: `MinMaxScaler`, `preprocessing` don't change
- **pandas/numpy stay the same**: Data loading and manipulation unchanged
- **Stateful LSTM**: PyTorch handles this by passing hidden state between calls
- **GPU usage**: Add `.cuda()` calls or use `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`

---

## Quick Start After Migration

```python
# Check GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FantasyLSTM().to(device)
```
