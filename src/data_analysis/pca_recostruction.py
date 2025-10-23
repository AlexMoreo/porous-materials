import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from data import *
from utils import *
from tqdm import tqdm

reduce='input'  # 8 dimenions
# reduce='nitrogen'
# reduce='hydrogen'

components=8

path_nitrogen = f'../../data/training/dataset_for_nitrogen.csv'
path_hydrogen = f'../../data/training/dataset_for_hydrogen.csv'

if reduce=='input':
    X, _ = load_data(path_nitrogen, cumulate_x=True, normalize=True)
    Z = X
elif reduce=='nitrogen':
    _, y = load_data(path_nitrogen, cumulate_x=True, normalize=True)
    Z = y
elif reduce=='hydrogen':
    _, y = load_data(path_hydrogen, cumulate_x=True, normalize=True)
    Z = y


errors = []
loo = LeaveOneOut()
for i, (train, test) in tqdm(enumerate(loo.split(Z)), desc='generating reconstructions', total=len(Z)):
    test_name = f'model{i+1}'

    Ztr = Z[train]
    Zte = Z[test]

    pca = PCA(n_components=components)
    pca.fit(Ztr)
    Zte_ = pca.transform(Zte)
    Zte_ = pca.inverse_transform(Zte_)

    plot_result(Zte[0], Zte_[0], f'../reconstruction/{reduce}/PCA{components}/{str(test_name)}.png', err_fun=mse)
    errors.append(mse(Zte[0], Zte_[0]))

print(f'Process ended. Reconstruction with {components=} has mse={np.mean(errors)}')


