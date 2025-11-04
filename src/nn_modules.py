import torch
from torch import nn
from torch.nn import functional as F
import math


def sequential(input_dim, output_dim, hidden, activation, dropout):
    layers = []
    in_features = input_dim
    for hidden_size in hidden:
        layers.append(nn.Linear(in_features, hidden_size))
        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        in_features = hidden_size

    layers.append(nn.Linear(in_features, output_dim))
    return nn.Sequential(*layers)


class FF3W(nn.Module):
    def __init__(self, Xdim, Zdim, Ydim, Ldim, hidden, activation=nn.ReLU, dropout=0):
        super(FF3W, self).__init__()
        self.Zdim = Zdim

        self.X2ZL_branch = sequential(Xdim, Zdim+Ldim, hidden, activation, dropout)
        self.ZL2X_branch = sequential(Zdim+Ldim, Xdim, hidden, activation, dropout)
        self.ZL2Y_branch = sequential(Zdim+Ldim, Ydim, hidden, activation, dropout)

    def forward(self, X):
        ZL = self.X2ZL_branch(X)
        Xrec = self.ZL2X_branch(ZL)
        Ypred = self.ZL2Y_branch(ZL)
        Zrec = ZL[:,:self.Zdim]
        return Xrec, Zrec, Ypred


class AE(nn.Module):
    def __init__(self, Xdim, Zdim, Ydim, Ldim, hidden, activation=nn.ReLU, dropout=0):
        super(AE, self).__init__()
        self.Zdim = Zdim
        self.dropout = dropout

        self.X2ZL_branch = sequential(Xdim, Zdim+Ldim, hidden, activation, dropout)
        self.ZL2Y_branch = sequential(Zdim+Ldim, Ydim, hidden, activation, dropout)

    def forward(self, X):
        ZL = self.X2ZL_branch(X)
        Zrec = ZL[:,:self.Zdim]
        hidden = F.relu(ZL)
        Ypred = self.ZL2Y_branch(hidden)
        return None, Zrec, Ypred


class AE2(nn.Module):
    def __init__(self, Xdim, Zdim, Ydim, Ldim, hidden, activation=nn.ReLU, dropout=0):
        super(AE2, self).__init__()
        self.Zdim = Zdim
        self.dropout = dropout
        self.Zproj = nn.Linear(Ldim, Zdim)

        self.X2L_branch = sequential(Xdim, Ldim, hidden, activation, dropout)
        self.L2Y_branch = sequential(Ldim, Ydim, hidden, activation, dropout)

    def forward(self, X):
        L = self.X2L_branch(X)
        hidden = F.relu(L)
        Zrec = self.Zproj(hidden)
        Ypred = self.L2Y_branch(hidden)
        return None, Zrec, Ypred


class AE3(nn.Module):
    def __init__(self, Xdim, Zdim, Ydim, Ldim, hidden, activation=nn.ReLU, dropout=0):
        super(AE3, self).__init__()
        self.Zdim = Zdim
        self.dropout = dropout

        self.X2L_branch = sequential(Xdim, Ldim, hidden, activation, dropout)
        self.L2Y_branch = sequential(Ldim, Ydim, hidden, activation, dropout)
        self.L2Z_branch = sequential(Ldim, Zdim, hidden, activation, dropout)

    def forward(self, X):
        L = self.X2L_branch(X)
        hidden = F.relu(L)
        Zrec = self.L2Z_branch(hidden)
        Ypred = self.L2Y_branch(hidden)
        return None, Zrec, Ypred






