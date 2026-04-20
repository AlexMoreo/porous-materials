import os

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn, optim
from os.path import join
from sklearn.decomposition import PCA
import joblib
from pathlib import Path


class PCAadapt:

    def __init__(self, components, normalize=False):
        self.n_components = components
        self.normalize = normalize
        self.pca = PCA(n_components=self.n_components)
        self.minmaxscaler = MinMaxScaler()

    def _compress(self, Z, fit, null_val=None):
        null_mask = (Z==null_val)
        null_rows = null_mask.all(axis=1)

        nel, ndim = Z.shape
        Z_valid = Z[~null_rows]

        # normalization step
        if self.normalize:
            unsafe_values_mask = null_mask[~null_rows]
            if fit:
                norm_fn = self.minmaxscaler.fit_transform
                # avoid null_val to be taken as the min for the scaling

                safe_value = Z_valid[~unsafe_values_mask].mean()
                Z_valid[unsafe_values_mask] = safe_value
            else:
                norm_fn = self.minmaxscaler.transform
            Z_valid = norm_fn(Z_valid)
            Z_valid[unsafe_values_mask] = null_val  # restores the null_value

        # PCA (conditional)
        if self.n_components is not None:
            if ndim > self.n_components:
                compress_fn = self.pca.fit_transform if fit else self.pca.transform
                Z_valid = compress_fn(Z_valid)

        # reconstruct for null values
        Z_out = np.full((nel, Z_valid.shape[1]), null_val, dtype=np.float32)
        Z_out[~null_rows] = Z_valid

        return Z_out

    def fit_transform(self, Z, null_val=None):
        self.orig_dim = Z.shape[1]
        return self._compress(Z, fit=True, null_val=null_val)

    def transform(self, Z, Z_null=None):
        return self._compress(Z, fit=False, null_val=Z_null)

    def inverse_transform(self, Z):
        if self.n_components is not None:
            if Z.shape[1] != self.n_components:
                raise ValueError('inverse transform not understood')

            if self.orig_dim > self.n_components:
                Z = self.pca.inverse_transform(Z)

        if self.normalize:
            Z = self.minmaxscaler.inverse_transform(Z)

        return Z

class NeuralRegressor:
    def __init__(self,
                 model,
                 lr=0.003,
                 cuda=True,
                 X_red=None, Y_red=None, Z_red=None,
                 wX=1, wY=1, wZ=1,
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
        self.cuda = cuda
        self.adapt_X = PCAadapt(components=X_red, normalize=normalize_XYZ)
        self.adapt_Y = PCAadapt(components=Y_red, normalize=normalize_XYZ)
        self.adapt_Z = PCAadapt(components=Z_red, normalize=normalize_XYZ)
        self.wX = wX
        self.wY = wY
        self.wZ = wZ
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

    def fit(self, X, Y, Z, in_val=None):

        Y_mask = (Y != -1)
        if not self.allow_incomplete_Y:
            complete_mask = Y_mask.all(axis=1)  # indicates fully specified rows in Y
            print(f'filtering out {(1-complete_mask).sum()} incompletely specified samples')
            X = X[complete_mask]
            Y = Y[complete_mask]
            Z = Z[complete_mask]
            Y_mask = None

        Z_null = full_null_rows(Z)
        if not self.allow_incomplete_Z:
            X = X[~Z_null]
            Y = Y[~Z_null]
            Z = Z[~Z_null]
            if Y_mask is not None:
                Y_mask = Y_mask[~Z_null]
            Z_null = np.full(shape=Z.shape[0], fill_value=False, dtype=bool)

        X = self.adapt_X.fit_transform(X)
        Xtr, Xval = split_to_tensors(X, in_val, device=self.device)

        Y = self.adapt_Y.fit_transform(Y, null_val=-1)
        Ytr, Yval = split_to_tensors(Y, in_val, device=self.device)
        Ytr_mask, Yval_mask = split_to_tensors(Y_mask, in_val, dtype=torch.bool, device=self.device)

        Z = self.adapt_Z.fit_transform(Z, null_val=-1)
        Ztr, Zval = split_to_tensors(Z, in_val, device=self.device)
        Ztr_null, Zval_null = split_to_tensors(Z_null, in_val, dtype=torch.bool)

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
            loss_tr, loss_y = self.loss_fn(Xtr, X_recons, Ytr, Y_predicted, Ztr, Z_recons, Ztr_null, Ytr_mask)
            loss_tr.backward()
            optimizer.step()

            # validation step
            # ---------------
            if in_val is not None:
                self.model.eval()
                X_recons, Z_recons, Y_predicted = self.model(Xval)
                loss_val, loss_y_val = self.loss_fn(Xval, X_recons, Yval, Y_predicted, Zval, Z_recons, Zval_null, Yval_mask)
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
        self.model.load_state_dict(torch.load(best_model_path, map_location=torch.device(self.device)))
        self.best_loss = best_loss

        # save the dimensionality reduction objects
        joblib.dump(self.adapt_X, join(self.checkpoint_dir, f'{self.checkpoint_id}_adaptX.pkl'))
        joblib.dump(self.adapt_Y, join(self.checkpoint_dir, f'{self.checkpoint_id}_adaptY.pkl'))
        joblib.dump(self.adapt_Z, join(self.checkpoint_dir, f'{self.checkpoint_id}_adaptZ.pkl'))

        return self

    def load_model(self, path, device='cpu'):
        self.device = device
        self.model.to(torch.device(device))
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
        return loss, y_loss

    def predict(self, X, return_XZ=False):
        X = self.adapt_X.transform(X)
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            X_recons, Z_recons, Y_predicted = self.model(X_tensor)
            Y_predicted = Y_predicted.detach().cpu().numpy()
            Y_predicted = self.adapt_Y.inverse_transform(Y_predicted)
            if return_XZ:
                if X_recons is not None:
                    X_recons = X_recons.detach().cpu().numpy()
                    X_recons = self.adapt_X.inverse_transform(X_recons)

                Z_recons = Z_recons.detach().cpu().numpy()
                Z_recons = self.adapt_Z.inverse_transform(Z_recons)
                return Y_predicted, X_recons, Z_recons
            return Y_predicted


def full_null_rows(Z):
    Z_null = np.all(Z == -1, axis=1)
    return Z_null


def masked_mse(Y_hat, Y, Y_mask=None):
    """
    Allows for partially unspecified values in Y; these are assumed to be represented by "-1" values

    :param Y_hat: predicted values
    :param Y: true values; may contain missing values indicated by a "-1"
    :param Y_mask: boolean mask, indicating valid values
    :return: the MSE between Y_hat and Y, skipping missing values
    """
    if Y_mask is None:
        Y_mask = (Y!=-1)
    mse = (Y_hat - Y) ** 2
    mse = mse * Y_mask.float()
    return mse.sum() / (Y_mask.sum() + 1e-15)


def split_to_tensors(W, in_val, dtype=torch.float32, device='cpu'):
    if W is None:
        return None, None

    if in_val is not None:
        W_tr = torch.tensor(W[~in_val], dtype=dtype, device=device)
        W_val = torch.tensor(W[in_val], dtype=dtype, device=device)
    else:
        W_tr = torch.tensor(W, dtype=dtype, device=device)
        W_val = None
    return W_tr, W_val


def closest_to_mean(curves):
    curves = np.asarray(curves)
    mean_curve = curves.mean(axis=0)
    diff = np.mean((curves - mean_curve) ** 2, axis=1)
    smallest_pos = np.argmin(diff)
    return curves[smallest_pos]
