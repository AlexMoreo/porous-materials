from abc import ABC, abstractmethod

import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR
import torch
from torch import nn, optim


class StackRegressor:
    def __init__(self):
        self.first_tier = MultiOutputRegressor(LinearSVR())
        self.second_tier = MultiOutputRegressor(LinearSVR())

    def fit(self, X, y):
        loo = LeaveOneOut()
        Z = []
        for train, test in loo.split(X, y):
            Xtr, ytr = X[train], y[train]
            Xte, yte = X[test], y[test]

            self.first_tier.fit(Xtr, ytr)
            yte_pred = self.first_tier.predict(Xte)
            Z.append(yte_pred[0])

        self.first_tier.fit(X, y)
        self.second_tier.fit(Z, y)

    def predict(self, X):
        Z = self.first_tier.predict(X)
        y_hat = self.second_tier.predict(Z)
        return y_hat


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output



class LSTMregressor:
    def __init__(self, hidden_size=128, num_layers=3):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model = None

    def fit(self, X, y):
        input_size = X.shape[1]
        output_size = y.shape[1]
        self.model = LSTMModel(input_size, hidden_size=self.hidden_size, output_size=output_size, num_layers=self.num_layers)

        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        num_epochs = 1000
        PATIENCE = 25
        val_loss = np.inf
        patience = PATIENCE

        for epoch in range(num_epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

            if loss < val_loss:
                val_loss=loss
                patience=PATIENCE
            else:
                patience-=1

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.10f}')
            if patience<=0:
                print(f'Method stopped after {epoch} epochs')
                break

        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
            ypred = self.model(X)
            ypred = ypred.detach().cpu().numpy()
            return ypred