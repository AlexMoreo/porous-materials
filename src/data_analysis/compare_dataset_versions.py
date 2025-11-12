from data import load_both_data

path_h2 = '../../data/training/dataset_for_hydrogen.csv'
path_n2 = '../../data/training/dataset_for_nitrogen.csv'
test_names_v1, Vin_v1, Gin_v1, Gout_v1 =  load_both_data(
    path_input_gas=path_n2, path_output_gas=path_h2, cumulate_vol=True, normalize=True,
    return_index=True, exclude_id=['model41', 'model45']
)

path_h2 = '../../data/training_molg/dataset_hydrogen_molg.csv'
path_n2 = '../../data/training_molg/dataset_nitrogen_molg.csv'
test_names_v2, Vin_v2, Gin_v2, Gout_v2 = load_both_data(
    path_input_gas=path_n2, path_output_gas=path_h2, cumulate_vol=False, normalize=True,
    return_index=True, exclude_id=['model41', 'model45']
)

print('loaded')