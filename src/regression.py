import os

import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
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

    def __init__(self, components, normalize=False):
        self.n_components = components
        self.normalize = normalize
        self.pca = PCA(n_components=self.n_components)
        self.minmaxscaler = MinMaxScaler()

    def _compress(self, Z, fit, Z_null=None):
        if Z_null is None:
            Z_null = np.full(shape=Z.shape[0], fill_value=False)

        nel, ndim = Z.shape
        Z_valid = Z[~Z_null]

        # normalization step
        if self.normalize:
            norm_fn = self.minmaxscaler.fit_transform if fit else self.minmaxscaler.transform
            Z_valid = norm_fn(Z_valid)

        # PCA (conditional)
        if self.n_components is not None:
            if ndim > self.n_components:
                compress_fn = self.pca.fit_transform if fit else self.pca.transform
                Z_valid = compress_fn(Z_valid)

        # reconstruct for null values
        Z_out = np.full((nel, Z_valid.shape[1]), -1., dtype=np.float32)
        Z_out[~Z_null] = Z_valid

        return Z_out

    def fit_transform(self, Z, Z_null=None):
        self.orig_dim = Z.shape[1]
        return self._compress(Z, fit=True, Z_null=Z_null)

    def transform(self, Z, Z_null=None):
        return self._compress(Z, fit=False, Z_null=Z_null)

    def inverse_transform(self, Z):
        if self.n_components is not None:
            if Z.shape[1] != self.n_components:
                raise ValueError('inverse transform not understood')

            if self.orig_dim > self.n_components:
                Z = self.pca.inverse_transform(Z)

        if self.normalize:
            Z = self.minmaxscaler.inverse_transform(Z)

        return Z


def jaggedness(output):
    # Define second-derivative kernel
    kernel = torch.tensor([1.0, -2.0, 1.0], dtype=output.dtype, device=output.cuda).view(1, 1, -1)

    # Add channel dimension for conv1d
    conv_out = F.conv1d(output.unsqueeze(1), kernel, padding=0)

    # Remove channel dimension and compute mean squared
    penalty = torch.mean(conv_out ** 2)
    return penalty


class NN3WayReg:
    def __init__(self,
                 model,
                 lr=0.003,
                 smooth_reg_weight=0.,
                 clip=None,
                 cuda=True,
                 X_red=None, Y_red=None, Z_red=None,
                 wX=1, wY=1, wZ=1,
                 monotonic_Z=False,
                 smooth_prediction=False,
                 weight_decay=0,
                 allow_incomplete_Z=True,
                 allow_incomplete_Y=False,
                 normalize_XYZ=False,
                 checkpoint_dir='../checkpoints',
                 checkpoint_id=None,
                 max_epochs=np.inf):

        assert wY > 0, 'prediction weight cannot be 0'
        if checkpoint_id is None:
            checkpoint_id = np.random.randint(low=0, high=10e8)
        assert Y_red is None or not allow_incomplete_Y, \
            (f"{Y_red=} was specified (thus PCA will be applied on Y), but {allow_incomplete_Y=}; "
             f"this combination is incompatible")
        self.model = model
        self.lr = lr
        self.reg_strength = smooth_reg_weight
        self.clip = clip
        self.cuda = cuda
        self.adapt_X = PCAadapt(components=X_red, normalize=normalize_XYZ)
        self.adapt_Y = PCAadapt(components=Y_red, normalize=normalize_XYZ)
        self.adapt_Z = PCAadapt(components=Z_red, normalize=normalize_XYZ)
        self.wX = wX
        self.wY = wY
        self.wZ = wZ
        self.monotonic_Z = monotonic_Z
        self.smooth_prediction = smooth_prediction
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_id = checkpoint_id
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.allow_incomplete_Z = allow_incomplete_Z
        self.allow_incomplete_Y = allow_incomplete_Y

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
        def split_val(W, adapt_W, null=None):
            W = adapt_W.fit_transform(W, null)

            if in_val is not None:
                W_tr = torch.tensor(W[~in_val], dtype=torch.float32, device=self.device)
                W_val = torch.tensor(W[in_val], dtype=torch.float32, device=self.device)
                if null is not None:
                    W_tr_null = torch.tensor(null[~in_val], dtype=torch.bool, device=self.device)
                    W_val_null = torch.tensor(null[in_val], dtype=torch.bool, device=self.device)
                    return W_tr, W_val, W_tr_null, W_val_null
            else:
                W_tr = torch.tensor(W, dtype=torch.float32, device=self.device)
                W_val = None
                if null is not None:
                    W_tr_null = torch.tensor(null, dtype=torch.bool, device=self.device)
                    W_val_null = None
                    return W_tr, W_val, W_tr_null, W_val_null
            return W_tr, W_val

        Y_mask = (Y != -1)
        if not self.allow_incomplete_Y:
            complete_mask = Y_mask.all(axis=1)  # indicates fully specified rows in Y
            print(f'filtering out {(1-complete_mask).sum()} incompletely specified samples')
            X = X[complete_mask]
            Y = Y[complete_mask]
            Z = Z[complete_mask]
            Y_mask = None

        Z_null = null_positions(Z)
        if not self.allow_incomplete_Z:
            X = X[~Z_null]
            Y = Y[~Z_null]
            Z = Z[~Z_null]
            if Y_mask is not None:
                Y_mask = Y_mask[~Z_null]
            Z_null = np.full(shape=Z.shape[0], fill_value=False, dtype=bool)

        Xtr, Xval = split_val(X, self.adapt_X)
        Ytr, Yval = split_val(Y, self.adapt_Y)
        Ztr, Zval, Ztr_null, Zval_null = split_val(Z, self.adapt_Z, null=Z_null)

        if Y_mask is not None:
            Y_mask = torch.tensor(Y_mask, device=self.device)

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

        while patience > 0:
            # training step
            # -------------
            self.model.train()
            optimizer.zero_grad()
            X_recons, Z_recons, Y_predicted = self.model(Xtr)
            Y_predicted, Z_recons = self.post_process(Y_predicted, Z_recons)
            loss_tr, loss_y = self.loss_fn(Xtr, X_recons, Ytr, Y_predicted, Ztr, Z_recons, Ztr_null, Y_mask)
            loss_tr.backward()
            optimizer.step()

            # validation step
            # ---------------
            if in_val is not None:
                self.model.eval()
                X_recons, Z_recons, Y_predicted = self.model(Xval)
                Y_predicted, Z_recons = self.post_process(Y_predicted, Z_recons)
                loss_val, loss_y_val = self.loss_fn(Xval, X_recons, Yval, Y_predicted, Zval, Z_recons, Zval_null, Y_mask)
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
                print('\n[STOP] maximum number of epochs reached')
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

    def loss_fn(self, X, X_hat, Y, Y_hat, Z, Z_hat, Z_null, Y_mask=None):
        valid_mask = ~Z_null

        criterion = nn.MSELoss()
        if Y_mask is None:
            y_loss = criterion(Y_hat, Y)
        else:
            y_loss = masked_mse(Y_hat, Y, Y_mask)

        x_loss = 0 if self.wX == 0 else criterion(X_hat, X)
        z_loss = 0 if self.wZ == 0 else criterion(Z_hat[valid_mask], Z[valid_mask])
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


def null_positions(Z):
    Z_null = np.all(Z == -1, axis=1)
    return Z_null


def masked_mse(Y_hat, Y, Y_mask):
    """
    Allows for partially unspecified values in Y; these are assumed to be represented by "-1" values

    :param Y_hat: predicted values
    :param Y: true values; may contain missing values indicated by a "-1"
    :param Y_mask: boolean mask, indicating valid values
    :return: the MSE between Y_hat and Y, skipping missing values
    """
    mse = (Y_hat - Y) ** 2
    mse = mse * Y_mask.float()
    return mse.sum() / (Y_mask.sum() + 1e-8)


class DirectRegression:
    def __init__(self, regressor:BaseEstimator, x_red=None, y_red=None, normalize=False):
        self.adaptX = PCAadapt(components=x_red, normalize=normalize)
        self.adaptY = PCAadapt(components=y_red, normalize=normalize)
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
        self.adaptX = PCAadapt(components=x_red)
        self.adaptZ = PCAadapt(components=z_red)
        self.adaptY = PCAadapt(components=y_red)
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
