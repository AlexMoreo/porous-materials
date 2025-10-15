import os

import numpy as np
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR
import torch
from torch import nn, optim
from os.path import join
import torch.nn.functional as F
from sklearn.base import clone
from sklearn.decomposition import PCA


class PCAadapt:

    def __init__(self, components, force=True):
        self.n_components = components
        self.force = force
        self.pca = PCA(n_components=self.n_components)
        if force:
            assert components is not None, 'components cannot be None if force=True'

    def fit_transform(self, Z):
        if not self.force and self.n_components is None:
            return Z
        if Z.shape[1] > self.n_components:
            Z = self.pca.fit_transform(Z)
        elif self.force:
            raise ValueError(f'requested PCA components={self.n_components} but input has {Z.shape[1]} dimensions')
        return Z

    def transform(self, Z):
        if not self.force and self.n_components is None:
            return Z
        if Z.shape[1] > self.n_components:
            Z = self.pca.transform(Z)
        elif self.force:
            raise ValueError(f'requested PCA components={self.n_components} but input has {Z.shape[1]} dimensions')
        return Z

    def inverse_transform(self, Z):
        if not self.force and self.n_components is None:
            return Z
        if Z.shape[1] == self.n_components:
            return self.pca.inverse_transform(Z)
        else:
            raise ValueError('inverse transform not understood')


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

    def jaggedness(self, output):
        # Define second-derivative kernel
        kernel = torch.tensor([1.0, -2.0, 1.0], dtype=output.dtype, device=output.device).view(1, 1, -1)

        # Add channel dimension for conv1d
        conv_out = F.conv1d(output.unsqueeze(1), kernel, padding=0)

        # Remove channel dimension and compute mean squared
        penalty = torch.mean(conv_out ** 2)
        return penalty

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
                loss += self.reg_strength * self.jaggedness(outputs)
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


class NN3WayReg:
    def __init__(self, model, lr=0.003, reg_strength=0., clip=None, cuda=True,
                 reduce_X=None, reduce_Y=None, reduce_Z=None, checkpoint_dir='../checkpoints', checkpoint_id=0):
        self.model = model
        self.lr = lr
        self.reg_strength = reg_strength
        self.clip = clip
        self.cuda = cuda
        self.adapt_X = PCAadapt(components=reduce_X, force=False)
        self.adapt_Y = PCAadapt(components=reduce_Y, force=False)
        self.adapt_Z = PCAadapt(components=reduce_Z, force=False)
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_id = checkpoint_id

        if cuda:
            self.model.cuda()
            self.device = 'cuda'
        else:
            self.device = 'cpu'


    def fit(self, X, Y, Z):

        X = self.adapt_X.fit_transform(X)
        Y = self.adapt_X.fit_transform(Y)
        Z = self.adapt_X.fit_transform(Z)

        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        Y_tensor = torch.tensor(Y, dtype=torch.float32, device=self.device)
        Z_tensor = torch.tensor(Z, dtype=torch.float32, device=self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        num_epochs = np.inf
        PATIENCE = 5000
        best_loss = np.inf
        patience = PATIENCE
        epoch = 0

        # loss partial weights
        wX, wZ, wY = 1, 1, 1

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        best_model_path = join(self.checkpoint_dir, f"best_model_{self.checkpoint_id}.pt")

        #for epoch_ in range(num_epochs):
        while patience > 0:
            self.model.train()
            optimizer.zero_grad()
            X_recons, Z_recons, Y_predicted = self.model(X_tensor)
            if self.clip:
                Y_predicted = 1 - F.relu(1 - Y_predicted)

            loss = wX*criterion(X_recons, X_tensor) + wZ*criterion(Z_recons, Z_tensor) + wY*criterion(Y_predicted, Y_tensor)

            if self.reg_strength>0:
                loss += self.reg_strength * self.jaggedness(Y_predicted)

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

    def predict(self, X, return_XZ=False):
        X = self.adapt_X.transform(X)
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            X_recons, Z_recons, Y_predicted = self.model(X_tensor)
            if self.clip:
                ypred = torch.clamp(Y_predicted, min=0.0, max=1.0)
            ypred = ypred.detach().cpu().numpy()
            ypred = self.adapt_Y.inverse_transform(ypred)
            if return_XZ:
                X_recons = X_recons.detach().cpu().numpy()
                X_recons = self.adapt_X.inverse_transform(X_recons)

                Z_recons = Z_recons.detach().cpu().numpy()
                Z_recons = self.adapt_Z.inverse_transform(Z_recons)
                return ypred, X_recons, Z_recons
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
