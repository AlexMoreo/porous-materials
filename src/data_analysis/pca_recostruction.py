import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from data import *
from utils import *
from tqdm import tqdm

reduce='volume'
# reduce='nitrogen'
# reduce='hydrogen'

components=8

path_n2 = f'../../data/training/dataset_for_nitrogen.csv'
path_h2 = f'../../data/training/dataset_for_hydrogen.csv'
Vin, Gin, Gout = load_both_data(path_input_gas=path_n2, path_output_gas=path_h2, cumulate_vol=True, normalize=True)

Z = {
    'volume': Vin,
    'nitrogen': Gin,
    'hydrogen': Gout
}[reduce]

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

    plot_result(Zte[0], Zte_[0], f'../../reconstruction/{reduce}/PCA{components}/{str(test_name)}.png', err_fun=mse)
    errors.append(mse(Zte[0], Zte_[0]))

print(f'Process ended. Reconstruction with {components=} has mse={np.mean(errors)}')


