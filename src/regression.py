import os

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVR
import torch
from torch import nn, optim
from os.path import join
import torch.nn.functional as F
from sklearn.base import clone
from sklearn.decomposition import PCA

import utils


class PCAadapt:

    def __init__(self, components, force=True):
        self.n_components = components
        self.force = force
        self.pca = PCA(n_components=self.n_components)
        if force:
            assert components is not None, 'components cannot be None if force=True'

    def _compress(self, Z, compress_fn):
        if not self.force and self.n_components is None:
            return Z
        if Z.shape[1] > self.n_components:
            Z = compress_fn(Z)
        elif self.force:
            raise ValueError(f'requested PCA components={self.n_components} but input has {Z.shape[1]} dimensions')
        return Z

    def fit_transform(self, Z):
        self.orig_dim = Z.shape[1]
        return self._compress(Z, self.pca.fit_transform)

    def transform(self, Z):
        return self._compress(Z, self.pca.transform)

    def inverse_transform(self, Z):
        if not self.force and self.n_components is None:
            return Z
        if Z.shape[1] == self.n_components:
            if self.orig_dim > self.n_components:
                return self.pca.inverse_transform(Z)
            else:
                return Z
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
                 reduce_X=None, reduce_Y=None, reduce_Z=None,
                 wX=1, wY=1, wZ=1,
                 checkpoint_dir='../checkpoints', checkpoint_id=0, max_epochs=np.inf):
        self.model = model
        self.lr = lr
        self.reg_strength = reg_strength
        self.clip = clip
        self.cuda = cuda
        self.adapt_X = PCAadapt(components=reduce_X, force=False)
        self.adapt_Y = PCAadapt(components=reduce_Y, force=False)
        self.adapt_Z = PCAadapt(components=reduce_Z, force=False)
        assert wY > 0, 'prediction weight cannot be 0'
        self.wX = wX
        self.wY = wY
        self.wZ = wZ
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_id = checkpoint_id
        self.max_epochs = max_epochs

        if cuda:
            self.model.cuda()
            self.device = 'cuda'
        else:
            self.device = 'cpu'


    def fit(self, X, Y, Z, in_val=None):

        def split_val(W, adapt_W):
            W = adapt_W.fit_transform(W)
            if in_val is not None:
                W_tr = torch.tensor(W[~in_val], dtype=torch.float32, device=self.device)
                W_val = torch.tensor(W[in_val], dtype=torch.float32, device=self.device)
            else:
                W_tr = torch.tensor(W, dtype=torch.float32, device=self.device)
                W_val = None
            return W_tr, W_val

        Xtr, Xval = split_val(X, self.adapt_X)
        Ytr, Yval = split_val(Y, self.adapt_Y)
        Ztr, Zval = split_val(Z, self.adapt_Z)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        PATIENCE = 5000
        best_loss = np.inf
        patience = PATIENCE
        epoch = 0

        # loss partial weights
        wX, wZ, wY = self.wX, self.wZ, self.wY

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        best_model_path = join(self.checkpoint_dir, f"best_model_{self.checkpoint_id}.pt")

        #for epoch_ in range(num_epochs):
        while patience > 0:
            # training step
            # -------------
            self.model.train()
            optimizer.zero_grad()
            X_recons, Z_recons, Y_predicted = self.model(Xtr)
            if self.clip:
                Y_predicted = 1 - F.relu(1 - Y_predicted)

            y_loss = criterion(Y_predicted, Ytr)
            x_loss = 0 if wX==0 else criterion(X_recons, Xtr)
            z_loss = 0 if wZ==0 else criterion(Z_recons, Ztr)
            loss_tr = wY*y_loss + wX*x_loss + wZ*z_loss

            if self.reg_strength>0:
                loss_tr += self.reg_strength * self.jaggedness(Y_predicted)

            loss_tr.backward()
            optimizer.step()

            # validation step
            # ---------------
            if in_val is not None:
                self.model.eval()
                X_recons, Z_recons, Y_predicted = self.model(Xval)
                if self.clip:
                    Y_predicted = 1 - F.relu(1 - Y_predicted)
                loss_val = criterion(Y_predicted, Yval)
                losstype = 'Val'
            else:
                # no validation: monitor the training loss instead
                loss_val = y_loss
                losstype = 'Tr'

            # eval monitoring
            # ---------------
            if loss_val < best_loss:
                best_loss = loss_val.item()
                patience=PATIENCE
                # save best model
                torch.save(self.model.state_dict(), best_model_path)
            else:
                patience-=1

            print(f'\rEpoch [{epoch + 1:05d}{"" if self.max_epochs==np.inf else f"/{self.max_epochs}"}, '
                  f'P={patience:04d}], '
                  f'Tr-Loss: {loss_tr.item():.10f}, '
                  f'Best {losstype}-Loss: {best_loss:.10f}', end='', flush=True)
            if patience<=0:
                print(f'\nMethod stopped after {epoch} epochs')
                break
            epoch+=1
            if self.max_epochs-epoch==0:
                break

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
                Y_predicted = torch.clamp(Y_predicted, min=0.0, max=1.0)
            Y_predicted = Y_predicted.detach().cpu().numpy()
            Y_predicted = self.adapt_Y.inverse_transform(Y_predicted)
            if return_XZ:
                X_recons = X_recons.detach().cpu().numpy()
                X_recons = self.adapt_X.inverse_transform(X_recons)

                Z_recons = Z_recons.detach().cpu().numpy()
                Z_recons = self.adapt_Z.inverse_transform(Z_recons)
                return Y_predicted, X_recons, Z_recons
            return Y_predicted


class NN2I1OReg:
    def __init__(self, model, lr=0.003, reg_strength=0., clip=None, cuda=True,
                 reduce_X=None, reduce_Y=None, reduce_Z=None,
                 wX=1, wY=1,
                 checkpoint_dir='../checkpoints', checkpoint_id=42):
        self.model = model
        self.lr = lr
        self.reg_strength = reg_strength
        self.clip = clip
        self.cuda = cuda
        self.adapt_X = PCAadapt(components=reduce_X, force=False)
        self.adapt_Y = PCAadapt(components=reduce_Y, force=False)
        self.adapt_Z = PCAadapt(components=reduce_Z, force=False)
        assert wY > 0, 'prediction weight cannot be 0'
        self.wX = wX
        self.wY = wY
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_id = checkpoint_id

        if cuda:
            self.model.cuda()
            self.device = 'cuda'
        else:
            self.device = 'cpu'


    def fit(self, X, Y, Z):

        X = self.adapt_X.fit_transform(X)
        Y = self.adapt_Y.fit_transform(Y)
        Z = self.adapt_Z.fit_transform(Z)

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
        wX, wY = self.wX, self.wY

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        best_model_path = join(self.checkpoint_dir, f"best_model_{self.checkpoint_id}.pt")

        #for epoch_ in range(num_epochs):
        while patience > 0:
            self.model.train()
            optimizer.zero_grad()
            X_recons, Y_predicted = self.model(X_tensor, Z_tensor)
            if self.clip:
                Y_predicted = 1 - F.relu(1 - Y_predicted)

            y_loss = criterion(Y_predicted, Y_tensor)
            x_loss = 0 if wX==0 else criterion(X_recons, X_tensor)
            loss = wY*y_loss + wX*x_loss

            if self.reg_strength>0:
                loss += self.reg_strength * self.jaggedness(Y_predicted)

            loss.backward()
            optimizer.step()

            if loss < best_loss:
                best_loss = loss.item()
                best_loss_y = y_loss.item()
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
            if num_epochs-epoch==0:
                break

        print()

        # Load the best model weights before returning
        self.model.load_state_dict(torch.load(best_model_path))
        self.best_loss = best_loss
        self.best_loss_y = best_loss_y

        return self

    def predict(self, X, Z, return_X=False):
        X = self.adapt_X.transform(X)
        Z = self.adapt_Z.transform(Z)
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            Z_tensor = torch.tensor(Z, dtype=torch.float32, device=self.device)
            X_recons, Y_predicted = self.model(X_tensor, Z_tensor)
            if self.clip:
                Y_predicted = torch.clamp(Y_predicted, min=0.0, max=1.0)
            Y_predicted = Y_predicted.detach().cpu().numpy()
            Y_predicted = self.adapt_Y.inverse_transform(Y_predicted)
            if return_X:
                X_recons = X_recons.detach().cpu().numpy()
                X_recons = self.adapt_X.inverse_transform(X_recons)

                return Y_predicted, X_recons
            return Y_predicted


class RandomForestXYPCA:
    def __init__(self, Xreduce_to, Yreduce_to):
        # super().__init__()
        self.adaptX = PCAadapt(components=Xreduce_to, force=False)
        self.adaptY = PCAadapt(components=Yreduce_to, force=False)
        self.rf = RandomForestRegressor()

    def fit(self, X, y, sample_weight=None):
        X = self.adaptX.fit_transform(X)
        y = self.adaptY.fit_transform(y)
        return self.rf.fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        X = self.adaptX.transform(X)
        y = self.rf.predict(X)
        return self.adaptY.inverse_transform(y)


class RandomForestXZYPCA:
    def __init__(self, Xreduce_to, Zreduce_to, Yreduce_to):
        # super().__init__()
        self.adaptX = PCAadapt(components=Xreduce_to, force=False)
        self.adaptZ = PCAadapt(components=Zreduce_to, force=False)
        self.adaptY = PCAadapt(components=Yreduce_to, force=False)
        self.rf = RandomForestRegressor()

    def fit(self, X, Z, y, sample_weight=None):
        X = self.adaptX.fit_transform(X)
        Z = self.adaptZ.fit_transform(Z)
        XZ = np.hstack([X,Z])
        y = self.adaptY.fit_transform(y)
        self.rf.fit(XZ, y, sample_weight=sample_weight)
        return self

    def predict(self, X, Z):
        X = self.adaptX.transform(X)
        Z = self.adaptZ.transform(Z)
        XZ = np.hstack([X, Z])
        y = self.rf.predict(XZ)
        return self.adaptY.inverse_transform(y)


class NearestNeighbor:
    def __init__(self):
        pass

    def fit(self, X, Y):
        self.nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        self.nn.fit(X)
        self.Y = Y

    def predict(self, X):
        distances, indices = self.nn.kneighbors(X)
        nearest = indices[:,0]
        return self.Y[nearest]


class MetaLearner:
    def __init__(self, base_methods, prediction_dir, X, Xnames):
        self.base_methods = base_methods
        self.prediction_dir = prediction_dir
        self.results = {}
        for method in self.base_methods:
            self.results[method] = utils.ResultTracker(join(self.prediction_dir, f'{method}.pkl')).results
        self.X = X
        self.Xnames = list(Xnames)

    def predict(self, test_name):
        available_predictions = None
        for method in self.base_methods:
            r = self.results[method][test_name]
            if available_predictions is None:
                available_predictions = r['test_names']
            else:
                assert set(available_predictions) == set(r['test_names'])

        X_available = []
        for available_pred in available_predictions:
            index = self.Xnames.index(available_pred)
            Xi = self.X[index]
            X_available.append(Xi)
        X_available = np.asarray(X_available)
        nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn.fit(X_available)

        test_index = self.Xnames.index(test_name)
        _, closest_idx = nn.kneighbors(self.X[test_index:test_index+1])
        closest_idx = closest_idx[:,0].item()
        closest_position = available_predictions[closest_idx]

        smallest_err = np.inf
        best_method = None
        for method in self.base_methods:
            r = self.results[method][test_name]
            err = r['errors'][closest_idx]
            if err < smallest_err:
                smallest_err = err
                best_method = method

        return best_method

