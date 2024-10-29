from abc import ABC, abstractmethod
from unicodedata import bidirectional

import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR
import torch
from torch import nn, optim
from os.path import join


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
    def __init__(self, input_size, hidden_size, output_size, num_layers, bidirectional):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size   , output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output


class FFModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, activation=nn.ReLU):
        """
        Parameters:
        - input_size (int): Number of input features.
        - hidden_sizes (list of int): List containing the number of units in each hidden layer.
        - output_size (int): Number of output units.
        - activation (nn.Module): Activation function to use (default: ReLU).
        """
        super(FFModel, self).__init__()

        layers = []
        in_features = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(activation())
            in_features = hidden_size

        layers.append(nn.Linear(in_features, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class NeuralRegressor:
    def __init__(self, model, lr=0.003, reg_strength=0.1):
        self.model = model
        self.lr = lr
        self.reg_strength = reg_strength
        self.model = model

    def jaggedness(self, output):
        center = output[:, 1:-1]
        left = output[:, :-2]
        right = output[:, 2:]
        return torch.mean(((2*center-left)-right)**2)

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        num_epochs = 5000
        PATIENCE = 100
        val_loss = np.inf
        patience = PATIENCE

        for epoch in range(num_epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor) + self.reg_strength * self.jaggedness(outputs)
            loss.backward()
            optimizer.step()

            if loss < val_loss:
                val_loss=loss
                patience=PATIENCE
            else:
                patience-=1

            if True or epoch%100==0:
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


class LSTMRegressor(NeuralRegressor):
    def __init__(self, input_size, output_size, hidden_size, num_layers, bidirectional, lr=0.003, reg_strength=0.1):
        model = LSTMModel(input_size, hidden_size=hidden_size, output_size=output_size,
                           num_layers=num_layers, bidirectional=bidirectional)
        super().__init__(model, lr=lr, reg_strength=reg_strength)


class TheirBaseline:
    def __init__(self, path, scale, basedir='../results/theirs'):
        self.basedir = basedir
        self.scale_inv = 1./scale
        self.path = path

    def fit(self, X, y):
        X, y = [], []
        prediction_file = join(self.basedir, self.path)
        prediction_file = prediction_file.replace('.csv', '_predicted_ads')
        with open(prediction_file, 'rt') as fin:
            lines = fin.readlines()
            for line in lines:
                Xi, yi = line.strip().split()
                Xi = float(Xi)
                yi = float(yi)
                X.append(Xi)
                y.append(yi)
        self.X = np.asarray(X)
        self.y = np.asarray(y).reshape(1, -1)

    def predict(self, X):
        # assert np.isclose(X, self.X).all(), 'wrong values'
        return self.y * self.scale_inv
