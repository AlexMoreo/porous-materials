import os

import numpy as np
from sklearn.model_selection import LeaveOneOut, KFold, cross_val_predict
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVR
import torch
from torch import nn, optim
from os.path import join
import torch.nn.functional as F
from sklearn.base import clone, BaseEstimator
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import utils
import time
import joblib
from pathlib import Path


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


def jaggedness(output):
    # Define second-derivative kernel
    kernel = torch.tensor([1.0, -2.0, 1.0], dtype=output.dtype, device=output.device).view(1, 1, -1)

    # Add channel dimension for conv1d
    conv_out = F.conv1d(output.unsqueeze(1), kernel, padding=0)

    # Remove channel dimension and compute mean squared
    penalty = torch.mean(conv_out ** 2)
    return penalty


class NN3WayReg:
    def __init__(self, model, lr=0.003, smooth_reg_weight=0., clip=None, cuda=True,
                 X_red=None, Y_red=None, Z_red=None,
                 wX=1, wY=1, wZ=1,
                 monotonic_Z=False,
                 smooth_prediction=False,
                 weight_decay=0,
                 checkpoint_dir='../checkpoints', checkpoint_id=None, max_epochs=np.inf):
        self.model = model
        self.lr = lr
        self.reg_strength = smooth_reg_weight
        self.clip = clip
        self.cuda = cuda
        self.adapt_X = PCAadapt(components=X_red, force=False)
        self.adapt_Y = PCAadapt(components=Y_red, force=False)
        self.adapt_Z = PCAadapt(components=Z_red, force=False)
        assert wY > 0, 'prediction weight cannot be 0'
        self.wX = wX
        self.wY = wY
        self.wZ = wZ
        self.monotonic_Z = monotonic_Z
        self.smooth_prediction = smooth_prediction
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_id is None:
            checkpoint_id = np.random.randint(low=0, high=10e8)
        self.checkpoint_id = checkpoint_id
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay

        if cuda:
            self.model.cuda()
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        if self.smooth_prediction:
            self.smooth_length = 5
            self.smooth_layer = nn.Conv1d(1, 1, kernel_size=self.smooth_length, padding=0, bias=False)
            self.smooth_layer.weight.data.fill_(1 / self.smooth_length)

    def fit(self, X, Y, Z, in_val=None, show_test_model=None):
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

        self.show_test_model = show_test_model
        if self.show_test_model is not None:
            Xte, Yte = self.show_test_model
            Xte = torch.tensor(self.adapt_X.transform(Xte), dtype=torch.float32, device=self.device)
            Yte = self.adapt_Y.transform(Yte)
            self.show_test_model = (Xte, Yte)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        PATIENCE = 5000
        best_loss = np.inf
        patience = PATIENCE
        epoch = 0

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        best_model_path = join(self.checkpoint_dir, f"best_model_{self.checkpoint_id}.pt")

        #for epoch_ in range(num_epochs):
        while patience > 0:
            # training step
            # -------------
            self.model.train()
            optimizer.zero_grad()
            X_recons, Z_recons, Y_predicted = self.model(Xtr)
            Y_predicted, Z_recons = self.post_process(Y_predicted, Z_recons)
            loss_tr, loss_y = self.loss_fn(Xtr, X_recons, Ytr, Y_predicted, Ztr, Z_recons)
            loss_tr.backward()
            optimizer.step()

            # validation step
            # ---------------
            if in_val is not None:
                self.model.eval()
                X_recons, Z_recons, Y_predicted = self.model(Xval)
                Y_predicted, Z_recons = self.post_process(Y_predicted, Z_recons)
                loss_val, loss_y_val = self.loss_fn(Xval, X_recons, Yval, Y_predicted, Zval, Z_recons)
                losstype = 'Val'
            else:
                # no validation: monitor the training loss instead
                loss_val = loss_tr
                losstype = 'Tr'

            # eval monitoring
            # ---------------
            if loss_val < best_loss:
                best_loss = loss_val.item()
                patience=PATIENCE
                # save best model
                torch.save(self.model.state_dict(), best_model_path)
                self.show_prediction_curve(self.show_test_model, epoch)
            else:
                patience-=1

            print(f'\rEpoch [{epoch + 1:05d}{"" if self.max_epochs==np.inf else f"/{self.max_epochs}"}, '
                  f'P={patience:04d}], '
                  f'Tr-Loss: {loss_tr.item():.10f}, Y-loss: {loss_y.item():.10f}, '
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

        # save the dimensionality reduction objects
        joblib.dump(self.adapt_X, join(self.checkpoint_dir, f'{self.checkpoint_id}_adaptX.pkl'))
        joblib.dump(self.adapt_Y, join(self.checkpoint_dir, f'{self.checkpoint_id}_adaptY.pkl'))
        joblib.dump(self.adapt_Z, join(self.checkpoint_dir, f'{self.checkpoint_id}_adaptZ.pkl'))

        return self

    def load_model(self, path, device='cpu'):
        self.model.load_state_dict(torch.load(path, map_location=torch.device(device)))
        parent = Path(path).parent
        self.adapt_X = joblib.load(join(parent, f'{self.checkpoint_id}_adaptX.pkl'))
        self.adapt_Y = joblib.load(join(parent, f'{self.checkpoint_id}_adaptY.pkl'))
        self.adapt_Z = joblib.load(join(parent, f'{self.checkpoint_id}_adaptZ.pkl'))
        return self

    def loss_fn(self, X, X_hat, Y, Y_hat, Z, Z_hat):
        criterion = nn.MSELoss()
        y_loss = criterion(Y_hat, Y)
        x_loss = 0 if self.wX == 0 else criterion(X_hat, X)
        z_loss = 0 if self.wZ == 0 else criterion(Z_hat, Z)
        loss = self.wY*y_loss + self.wX*x_loss + self.wZ*z_loss
        if self.reg_strength > 0:
            loss += self.reg_strength * jaggedness(Y_hat)
        return loss, y_loss

    def post_process(self, Y_hat, Z_hat):
        if self.clip:
            Y_hat = 1 - F.relu(1 - Y_hat)
        if self.monotonic_Z:
            Z_hat = self.monotonic(Z_hat)
        return Y_hat, Z_hat

    def post_smooth(self, output):
        if self.smooth_prediction:
            pad = self.smooth_length // 2
            output = torch.tensor(output, dtype=torch.float32).unsqueeze(1)
            output = F.pad(output, (pad, pad), mode='reflect')
            output = self.smooth_layer(output)
            return output.squeeze(1).detach().cpu().numpy()
        else:
            return output

    def monotonic(self, z):
        # Predicts increments only
        increments = F.relu(z)
        cumulative = torch.cumsum(increments, dim=1)  # guarantees monotonicity
        return cumulative

    def predict(self, X, return_XZ=False):
        X = self.adapt_X.transform(X)
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            X_recons, Z_recons, Y_predicted = self.model(X_tensor)
            Y_predicted, Z_recons = self.post_process(Y_predicted, Z_recons)
            # if self.clip:
            #     Y_predicted = torch.clamp(Y_predicted, min=0.0, max=1.0)
            Y_predicted = Y_predicted.detach().cpu().numpy()
            Y_predicted = self.adapt_Y.inverse_transform(Y_predicted)
            Y_predicted = self.post_smooth(Y_predicted)
            if return_XZ:
                if X_recons is not None:
                    X_recons = X_recons.detach().cpu().numpy()
                    X_recons = self.adapt_X.inverse_transform(X_recons)

                Z_recons = Z_recons.detach().cpu().numpy()
                Z_recons = self.adapt_Z.inverse_transform(Z_recons)
                Z_recons = self.post_smooth(Z_recons)
                return Y_predicted, X_recons, Z_recons
            return Y_predicted

    def show_prediction_curve(self, XY, epoch):

        if XY is not None:
            X, Y = XY
            self.model.eval()
            _, Z_recons, Y_predicted = self.model(X)
            Y_predicted, _ = self.post_process(Y_predicted, Z_recons)
            Y_predicted = Y_predicted.detach().cpu().numpy()
            Y_predicted = self.adapt_Y.inverse_transform(Y_predicted)
            Y_predicted_smooth = self.post_smooth(Y_predicted)

            Y = self.adapt_Y.inverse_transform(Y)

            # Z_recons = Z_recons.detach().cpu().numpy()
            # Z_recons = self.adapt_Z.inverse_transform(Z_recons)
            # Z_recons = self.post_smooth(Z_recons)

            if not hasattr(self, 'fig'):
                self.fig, self.ax = plt.subplots()
                plt.ion()
                x_axis = np.arange(Y_predicted.shape[1])
                self.ax.plot(x_axis, Y[0], label='Ground truth', color='black', linestyle='--')
                self.pred_line, = self.ax.plot(x_axis, np.zeros_like(x_axis), label='Prediction', color='red')
                self.predsmooth_line, = self.ax.plot(x_axis, np.zeros_like(x_axis), label='Smoothed Prediction', color='blue')
                self.epoch_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes, fontsize=12, color='blue')
                self.error_text = self.ax.text(0.02, 0.9, '', transform=self.ax.transAxes, fontsize=12, color='blue')
                self.ax.legend(loc='lower right')
                plt.show(block=False)

            self.pred_line.set_ydata(Y_predicted[0])
            self.predsmooth_line.set_ydata(Y_predicted_smooth[0])
            self.epoch_text.set_text(f"Epoch: {epoch + 1}")
            self.error_text.set_text(f"MSE: {utils.mse(Y, Y_predicted)*1e6:.5f}")

            # self.ax.set_ylim(pred_val.min() * 1.1, pred_val.max() * 1.1)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            time.sleep(0.001)




class DirectRegression:
    def __init__(self, regressor:BaseEstimator, x_red=None, y_red=None):
        self.adaptX = PCAadapt(components=x_red, force=False)
        self.adaptY = PCAadapt(components=y_red, force=False)
        self.rf = clone(regressor)

    def fit(self, X, Y, Z=None):
        assert Z is None, 'wrong value for Z, must be None'
        X = self.adaptX.fit_transform(X)
        Y = self.adaptY.fit_transform(Y)
        self.rf.fit(X, Y)
        return self

    def predict(self, X):
        X = self.adaptX.transform(X)
        y = self.rf.predict(X)
        return self.adaptY.inverse_transform(y)


class TwoStepRegression:
    def __init__(self, regressor:BaseEstimator, x_red=None, y_red=None, z_red=None):
        self.adaptX = PCAadapt(components=x_red, force=False)
        self.adaptZ = PCAadapt(components=z_red, force=False)
        self.adaptY = PCAadapt(components=y_red, force=False)
        self.rf1 = clone(regressor)
        self.rf2 = clone(regressor)

    def fit(self, X, Y, Z=None):
        X = self.adaptX.fit_transform(X)
        Z = self.adaptZ.fit_transform(Z)
        Y = self.adaptY.fit_transform(Y)
        Z_hat = cross_val_predict(self.rf1, X, Z, cv=5, n_jobs=5)
        self.rf1.fit(X, Z)
        self.rf2.fit(Z_hat, Y)
        return self

    def predict(self, X):
        X = self.adaptX.transform(X)
        Z_hat = self.rf1.predict(X)
        Y_hat = self.rf2.predict(Z_hat)
        return self.adaptY.inverse_transform(Y_hat)


class NearestNeighbor:
    def __init__(self):
        pass

    def fit(self, X, Y, Z=None):
        self.nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        self.nn.fit(X)
        self.Y = Y
        self.Z = Z

    def predict(self, X):
        distances, indices = self.nn.kneighbors(X)
        nearest = indices[:,0]
        if self.Z is not None:
            return self.Y[nearest], self.Z[nearest]
        else:
            return self.Y[nearest], None


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


def closest_to_mean(curves):
    curves = np.asarray(curves)
    mean_curve = curves.mean(axis=0)
    diff = np.mean((curves - mean_curve) ** 2, axis=1)
    smallest_pos = np.argmin(diff)
    return curves[smallest_pos]


def mean_curve(curves):
    curves = np.asarray(curves)
    return curves.mean(axis=0)


class Ensemble:
    def __init__(self, path, methods, agg='closest'):
        assert agg in ['closest', 'mean'], 'unexpected aggregation mode'
        self.path = path
        self.methods = methods
        self.results = {
            method: utils.ResultTracker(join(self.path, 'test_predictions', f'{method}.pkl')) for method in methods
        }
        self.agg = agg

    def predict(self, test_name):
        def aggregate_curve(curve):
            curves = []
            for method in self.methods:
                if test_name in self.results[method]:
                    curves.append(self.results[method].get(test_name)[curve][0])
            if len(curves) != len(self.methods):
                # incomplete aggregation
                return None
            if self.agg == 'closest':
                return closest_to_mean(curves)
            elif self.agg == 'mean':
                return mean_curve(curves)

        Gout = aggregate_curve('Gout')
        Vout = aggregate_curve('Vout')
        if Gout is None:
            return None, None
        return Gout.reshape(1,-1), Vout.reshape(1,-1)
