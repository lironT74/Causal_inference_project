import torch
import numpy as np
from MF_IPS_numpy_failed import get_numpy
import tqdm
import pandas as pd
from MF_IPS_torch import train_model_test, train_model_CV, find_best_key_dict
from sklearn.model_selection import KFold
from auxiliary import *

seed = 100

def get_inverse_propensities_smoothing(df_train_propensities, df_train, train_matrix, inter_coef, return_p_y_r=False):

    p_y_r = dict(df_train_propensities['rating'].value_counts() / len(df_train_propensities))
    p_y_r_o = dict(df_train['rating'].value_counts() / len(df_train))
    p_o_item = dict(df_train['song_id'].value_counts() / 15400)
    p_O_SUMS = dict(df_train['song_id'].value_counts())
    p_o = len(df_train) / train_matrix.size


    propensities = {(r, item): p_y_r_o[r]*(1-(inter_coef/(p_O_SUMS[item]+inter_coef))*p_o_item[item]+(inter_coef/(p_O_SUMS[item]+inter_coef))*p_o)*(1/p_y_r[r]) for r in p_y_r.keys() for item in p_o_item.keys()}
    for item in p_o_item.keys():
        propensities[(0, item)] = 0
    p_f = lambda r, item: 1/propensities[(r, item)] if propensities[(r,item)] != 0 else 0
    propensities_matrix = np.zeros((15400, 1000))
    for user in range(15400):
        for item in range(1000):
            r = train_matrix[user, item]
            propensities_matrix[user, item] = p_f(r, item+1)

    if return_p_y_r:
        return propensities_matrix, p_y_r
    return propensities_matrix


def read_yahoo_smoothing(path="data/yahoo_data", is_cv = False, inter_coef=0.5):
    column_names = ['user_id', 'song_id', 'rating']

    df_train = pd.read_csv(path+"/ydata-ymusic-rating-study-v1_0-train.txt", '\t', names=column_names)
    df_test_all = pd.read_csv(path+"/ydata-ymusic-rating-study-v1_0-test.txt", '\t', names=column_names)

    msk = np.random.rand(len(df_test_all)) < 0.95
    df_train_propensities = df_test_all[msk]
    df_test = df_test_all[~msk]

    train_matrix, test_matrix = get_numpy(df_train, df_test)

    inverse_propensities_matrix = get_inverse_propensities_smoothing(df_train_propensities, df_train, train_matrix, inter_coef)


    if is_cv:
        return df_train, df_test, df_train_propensities, train_matrix, test_matrix, inverse_propensities_matrix


    return train_matrix, test_matrix, inverse_propensities_matrix


def read_data_and_split_to_folds(iteration, delta_type=None, path="data/yahoo_data", k=4, inter_coef=0.5, path_to_save_txt="dirichlet_try"):


    df_train, df_test, df_train_propensities, Y, Y_test, inv_propensities = read_yahoo_smoothing(path, is_cv=True, inter_coef=inter_coef)

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    inner_dims = [5]
    lams = [1e-4, 1e-3, 1e-2, 1e-1, 1]

    mse_dict_total = {(lam, inner_dim): 0 for lam in lams for inner_dim in inner_dims}
    mae_dict_total = {(lam, inner_dim): 0 for lam in lams for inner_dim in inner_dims}

    num_users = 15400
    num_songs = 1000

    for fold_num, (train_index, val_index) in enumerate(kf.split(df_train)):

        fold_train_df = df_train.iloc[train_index]
        fold_val_df = df_train.iloc[val_index]

        Y_train = np.zeros((num_users, num_songs))
        Y_val = np.zeros((num_users, num_songs))

        for index, row in fold_train_df.iterrows():
            Y_train[row['user_id'] - 1, row['song_id'] - 1] = row['rating']

        for index, row in fold_val_df.iterrows():
            Y_val[row['user_id'] - 1, row['song_id'] - 1] = row['rating']


        train_propensities = get_inverse_propensities_smoothing(df_train_propensities, fold_train_df, Y_train, inter_coef=inter_coef)
        val_propensities = get_inverse_propensities_smoothing(df_train_propensities, fold_val_df, Y_val, inter_coef=inter_coef)

        train_propensities = train_propensities * (k / (k - 1))
        val_propensities = val_propensities * k

        if delta_type is None:

            mse_dict = train_model_CV(Y_train, Y_val, train_propensities, val_propensities, fold_num, iteration,
                                      delta_type="MSE", path_to_save_txt=path_to_save_txt)
            mae_dict = train_model_CV(Y_train, Y_val, train_propensities, val_propensities, fold_num, iteration,
                                      delta_type="MAE", path_to_save_txt=path_to_save_txt)

            for key in mse_dict:
                mse_dict_total[key] += mse_dict[key] / k
                mae_dict_total[key] += mae_dict[key] / k
        else:
            curr_dict = train_model_CV(Y_train, Y_val, train_propensities, val_propensities, fold_num, iteration,
                                      delta_type=delta_type, path_to_save_txt=path_to_save_txt)
            for key in curr_dict:
                mse_dict_total[key] += curr_dict[key] / k



    lam_mse, dim_mse = find_best_key_dict(mse_dict_total)
    lam_mae, dim_mae = find_best_key_dict(mae_dict_total)

    if delta_type is None:
        best_test_err_mse = train_model_test(Y, Y_test, inv_propensities, iteration, "MSE", dim_mse, lam_mse,
                                             path_to_save_txt=path_to_save_txt+'_test_error', EPOCHS=10)
        best_test_err_mae = train_model_test(Y, Y_test, inv_propensities, iteration, "MAE", dim_mae, lam_mae,
                                             path_to_save_txt=path_to_save_txt+'_test_error', EPOCHS=10)
    else:
        return train_model_test(Y, Y_test, inv_propensities, iteration, delta_type, dim_mse, lam_mse,
                                path_to_save_txt=path_to_save_txt+'_test_error', EPOCHS=10)

    return best_test_err_mse, best_test_err_mae


if __name__ == '__main__':
    np.random.seed(seed)
    torch.manual_seed(seed)
    err_list = []
    inter_coef = 5

    for i in range(5):
        print(f'START OF ITERATION {i + 1}')
        test_err = read_data_and_split_to_folds(i + 1, delta_type='MAE', path="data/yahoo_data", k=4,
                                                inter_coef=inter_coef, path_to_save_txt=f"MF-IPS mu={inter_coef}/dirichlet_try_mu_{inter_coef}")

    for i in range(5):
        print(f'START OF ITERATION {i + 1}')
        test_err = read_data_and_split_to_folds(i + 1, delta_type='MSE', path="data/yahoo_data", k=4,
                                                inter_coef=inter_coef, path_to_save_txt=f"MF-IPS mu={inter_coef}/dirichlet_try_mu_{inter_coef}")

    print_results(path=f'MF-IPS mu={inter_coef}/dirichlet_try_mu_5_MAE_CV.txt')
    print_results(path=f'MF-IPS mu={inter_coef}/dirichlet_try_mu_5_MSE_CV.txt')
