import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR
import torch
from torch import nn, optim
from os.path import join
import torch.nn.functional as F



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


class NeuralRegressor:
    def __init__(self, model, lr=0.003, reg_strength=0.1, clip=None, cuda=True):
        self.model = model
        self.lr = lr
        self.reg_strength = reg_strength
        self.clip = clip
        self.cuda = cuda

        if cuda:
            self.model.cuda()

    def jaggedness(self, output):
        # Define second-derivative kernel
        kernel = torch.tensor([1.0, -2.0, 1.0], dtype=output.dtype, device=output.device).view(1, 1, -1)

        # Add channel dimension for conv1d
        conv_out = F.conv1d(output.unsqueeze(1), kernel, padding=0)

        # Remove channel dimension and compute mean squared
        penalty = torch.mean(conv_out ** 2)
        return penalty

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        if self.cuda:
            X_tensor = X_tensor.cuda()
            y_tensor = y_tensor.cuda()

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        num_epochs = np.inf
        PATIENCE = 5000
        best_loss = np.inf
        patience = PATIENCE
        epoch = 0

        best_model_path = "best_model.pt"

        #for epoch_ in range(num_epochs):
        while patience > 0:
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            if self.clip:
                # outputs = F.sigmoid(outputs)
                outputs = 1-F.relu(1-outputs)
            loss = criterion(outputs, y_tensor) + self.reg_strength * self.jaggedness(outputs)
            loss.backward()
            optimizer.step()

            if loss < best_loss:
                best_loss=loss.item()
                patience=PATIENCE
                # save best model
                torch.save(self.model.state_dict(), best_model_path)
            else:
                patience-=1

            print(f'\rEpoch [{epoch + 1:05d}{"" if num_epochs==np.inf else f"/{num_epochs}"}, '
                  f'P={patience:04d}], '
                  f'Loss: {loss.item():.10f}, '
                  f'Best loss: {best_loss:.10f}', end='', flush=True)
            if patience<=0:
                print(f'\nMethod stopped after {epoch} epochs')
                break
            epoch+=1

        print()

        # Load the best model weights before returning
        self.model.load_state_dict(torch.load(best_model_path))
        self.best_loss = best_loss

        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32)
            if self.cuda:
                X = X.cuda()
            ypred = self.model(X)
            if self.clip:
                ypred = torch.clamp(ypred, min=0.0, max=1.0)
            ypred = ypred.detach().cpu().numpy()
            return ypred



class PrecomputedBaseline:
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
