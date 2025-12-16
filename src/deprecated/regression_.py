import numpy as np
import torch
from sklearn import clone
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR
from torch import nn, optim
from torch.nn import functional as F

from regression import PCAadapt, jaggedness


class NNReg:
    def __init__(self, model, lr=0.003, reg_strength=0.1, clip=None, cuda=True, reduce_in=None, reduce_out=None):
        self.model = model
        self.lr = lr
        self.reg_strength = reg_strength
        self.clip = clip
        self.cuda = cuda
        self.reduce_in = reduce_in
        self.reduce_out = reduce_out

        if cuda:
            self.model.cuda()

    # def __repr__(self):
    #     f'NR({self.model})-lr{self.lr}-rs{self.reg_strength}'

    def fit(self, X, y):

        if self.reduce_in:
            self.adapt_in = PCAadapt(components=self.reduce_in)
            X = self.adapt_in.fit_transform(X)
        if self.reduce_out:
            self.adapt_out = PCAadapt(components=self.reduce_out)
            y = self.adapt_out.fit_transform(y)

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
            self.model.train_idx()
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            if self.clip:
                # outputs = F.sigmoid(outputs)
                outputs = 1-F.relu(1-outputs)
            loss = criterion(outputs, y_tensor)
            if self.reg_strength>0:
                loss += self.reg_strength * jaggedness(outputs)
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
        if self.reduce_in:
            X = self.adapt_in.transform(X)

        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32)
            if self.cuda:
                X = X.cuda()
            ypred = self.model(X)
            if self.clip:
                ypred = torch.clamp(ypred, min=0.0, max=1.0)
            ypred = ypred.detach().cpu().numpy()
            if self.reduce_out:
                ypred = self.adapt_out.inverse_transform(ypred)
            return ypred


class StackRegressor:

    def __init__(self, regressor=LinearSVR(), mode='tat', k=5):
        self.first_tier = MultiOutputRegressor(clone(regressor))
        self.second_tier = MultiOutputRegressor(clone(regressor))
        self.mode = mode
        self.k = k
        assert mode in ['tat', 'loo', 'kfcv'], 'wrong mode'

    def fit(self, X, y):
        if self.mode in ['loo', 'kfcv']:
            if self.mode == 'loo':
                cv = LeaveOneOut()
            elif self.mode == 'kfcv':
                cv = KFold(n_splits=self.k)
            Z = []
            for train, test in cv.split(X, y):
                Xtr, ytr = X[train], y[train]
                Xte, yte = X[test], y[test]

                self.first_tier.fit(Xtr, ytr)
                yte_pred = self.first_tier.predict(Xte)
                Z.append(yte_pred)

            Z = np.vstack(Z)
            self.first_tier.fit(X, y)
            self.second_tier.fit(Z, y)
        elif self.mode == 'tat':
            self.first_tier.fit(X, y)
            Z = self.first_tier.predict(X)
            self.second_tier.fit(Z,y)


    def predict(self, X):
        Z = self.first_tier.predict(X)
        y_hat = self.second_tier.predict(Z)
        return y_hat
