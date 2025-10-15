import torch
from torch import nn
from torch.nn import functional as F
import math

class FF3W(nn.Module):
    def __init__(self, Xdim, Zdim, Ydim, Ldim, hidden, activation=nn.ReLU, dropout=0):
        super(FF3W, self).__init__()
        self.Zdim=Zdim

        self.X2ZL_branch = self.sequential(Xdim, Zdim+Ldim, hidden, activation, dropout)
        self.ZL2X_branch = self.sequential(Zdim+Ldim, Xdim, hidden, activation, dropout)
        self.ZL2Y_branch = self.sequential(Zdim+Ldim, Ydim, hidden, activation, dropout)

    def sequential(self, input_dim, output_dim, hidden, activation, dropout):
        layers = []
        in_features = input_dim
        for hidden_size in hidden:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(activation())
            layers.append(nn.Dropout(p=dropout))
            in_features = hidden_size

        layers.append(nn.Linear(in_features, output_dim))
        return nn.Sequential(*layers)

    def forward(self, X):
        ZL = self.X2ZL_branch(X)
        Xrec = self.ZL2X_branch(ZL)
        Ypred = self.ZL2Y_branch(ZL)
        Zrec = ZL[:,:self.Zdim]
        return Xrec, Zrec, Ypred


