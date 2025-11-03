from data import load_both_data

if __name__ == '__main__':
    path_h2 = '../data/training/dataset_for_hydrogen.csv'
    path_n2 = '../data/training/dataset_for_nitrogen.csv'
    test_names, Vin, Gin, Gout = load_both_data(
        path_input_gas=path_n2, path_output_gas=path_h2, cumulate_vol=True, normalize=True,
        return_index=True, exclude_id=['model41', 'model45']
    )

    'PAE2zy', NN3WayReg(
        model=AE2(
            Xdim=10, Zdim=10, Ydim=10, Ldim=1024, hidden=[1024]
        ), wX=0, wZ=0.001, X_red=10, Z_red=10, Y_red=10, lr=0.001,
        smooth_prediction=False, smooth_reg_weight=0.000, weight_decay=0.00000001, max_epochs=25_000
    )

    yield 'PAE2ZY', NN3WayReg(
        model=AE2(
            Xdim=Gi_dim, Zdim=V_dim, Ydim=Go_dim, Ldim=1024, hidden=[1024]
        ), wX=0, wZ=0.001, X_red=Gi_dim, Z_red=V_dim, Y_red=Go_dim, lr=0.001,
        smooth_prediction=False, smooth_reg_weight=0.000, weight_decay=0.00000001, max_epochs=50_000
    ),

    yield 'PAExy', NN3WayReg(  # nomenclature is wrong, should be PAEzy
        model=AE(
            Xdim=10, Zdim=10, Ydim=10, Ldim=1024, hidden=[1024]
        ), wX=0, wZ=0.0001, X_red=10, Z_red=10, Y_red=10, lr=0.001,
        smooth_prediction=False, smooth_reg_weight=0.000, weight_decay=0.00000001, max_epochs=25_000
    ),

    yield 'PAEZY', NN3WayReg(  # relauching... it was AE2 instead of AE
        model=AE(
            Xdim=Gi_dim, Zdim=V_dim, Ydim=Go_dim, Ldim=1024, hidden=[1024]
        ), wX=0, wZ=0.0001, X_red=Gi_dim, Z_red=V_dim, Y_red=Go_dim, lr=0.001,
        smooth_prediction=False, smooth_reg_weight=0.000, weight_decay=0.00000001, max_epochs=50_000
    ),