import pandas as pd
import pickle
import numpy as np
import tqdm
import torch.nn as nn
import torch
from scipy.sparse import dok_matrix
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans


seed = 100
inner_dims = [5]
lams = [1e-4, 1e-3, 1e-2, 1e-1, 1]
EPOCHS = 10


def df_to_dict(df, first_key, second_key, value_key):
    new_dict = {}
    for i in df[first_key].unique():
        new_dict[i] = [{df[second_key][j]: df[value_key][j]} for j in df[df[first_key] == i].index]

    return new_dict


def read_yahoo(path="data/yahoo_data"):
    column_names = ['user_id', 'song_id', 'rating']
    df_train = pd.read_csv(path+"/ydata-ymusic-rating-study-v1_0-train.txt", '\t', names=column_names)
    df_test = pd.read_csv(path+"/ydata-ymusic-rating-study-v1_0-test.txt", '\t', names=column_names)

    indexes = np.random.permutation(5400)
    df_test_test = df_test.iloc[indexes[:int(5400*0.95)]]
    df_train_test = df_test.iloc[indexes[int(5400*0.95):]]

    user2song_train = df_to_dict(df_train, 'user_id', 'song_id', 'rating')
    user2song_train_test = df_to_dict(df_train_test, 'user_id', 'song_id', 'rating')
    user2song_test_test = df_to_dict(df_test_test, 'user_id', 'song_id', 'rating')


    with open(path+'/user2song_train.pkl', 'wb') as f:
        pickle.dump(user2song_train, f)

    # with open(path+'/user2song_test.pkl', 'wb') as f:
    #     pickle.dump(user2song_test, f)
    with open(path+'/user2song_train_test.pkl', 'wb') as f:
        pickle.dump(user2song_train_test, f)

    with open(path+'/user2song_test_test.pkl', 'wb') as f:
        pickle.dump(user2song_test_test, f)


def load_yahoo(path="data/yahoo_data"):
    with open(path+'/user2song_train.pkl', 'rb') as f:
        user2song_train = pickle.load(f)

    Y_train = dok_matrix((15400, 1000), dtype=np.float32)
    for user_id in user2song_train:
        for song_raiting_pair in user2song_train[user_id]:
            song_id, raiting = list(song_raiting_pair.items())[0]
            Y_train[user_id-1, song_id-1] = raiting

    # with open(path+'/user2song_test.pkl', 'rb') as f:
    #     user2song_test = pickle.load(f)
    #
    # Y_test = np.zeros((5400, 1000))
    #
    # for user_id in user2song_test:
    #     for song_raiting_pair in user2song_test[user_id]:
    #         song_id, raiting = list(song_raiting_pair.items())[0]
    #         Y_test[user_id - 1, song_id - 1] = raiting

    with open(path+'/user2song_train_test.pkl', 'rb') as f:
        user2song_train_test = pickle.load(f)

    Y_train_test = dok_matrix((15400, 1000), dtype=np.float32)

    for user_id in user2song_train_test:
        for song_raiting_pair in user2song_train_test[user_id]:
            song_id, raiting = list(song_raiting_pair.items())[0]
            Y_train_test[user_id - 1, song_id - 1] = raiting

    with open(path+'/user2song_test_test.pkl', 'rb') as f:
        user2song_test_test = pickle.load(f)

    Y_test_test = dok_matrix((15400, 1000), dtype=np.float32)

    for user_id in user2song_test_test:
        for song_raiting_pair in user2song_test_test[user_id]:
            song_id, raiting = list(song_raiting_pair.items())[0]
            Y_test_test[user_id - 1, song_id - 1] = raiting
    # print(user2song_train)
    # return user2song_train, user2song_test, Y_train, Y_test
    return user2song_train, user2song_train_test, user2song_test_test, Y_train, Y_train_test, Y_test_test


def get_numpy(df_train, df_test):

    num_users = df_train['user_id'].nunique()
    num_songs = df_train['song_id'].nunique()

    train_matrix = np.zeros((num_users, num_songs))
    test_matrix = np.zeros((num_users, num_songs))

    for index, row in tqdm.tqdm(df_train.iterrows(), 'load MNAR'):
        train_matrix[row['user_id'] - 1, row['song_id'] - 1] = row['rating']

    for index, row in tqdm.tqdm(df_test.iterrows(), 'load MCAR'):
        test_matrix[row['user_id'] - 1, row['song_id'] - 1] = row['rating']

    return train_matrix, test_matrix


def find_best_key_dict(dict_total):
    return min(dict_total, key=dict_total.get)


def read_yahoo(get_inverse_propensities, path="data/yahoo_data", is_cv = False, *args, **kwargs):
    column_names = ['user_id', 'song_id', 'rating']

    df_train = pd.read_csv(path+"/ydata-ymusic-rating-study-v1_0-train.txt", '\t', names=column_names)
    df_test_all = pd.read_csv(path+"/ydata-ymusic-rating-study-v1_0-test.txt", '\t', names=column_names)

    msk = np.random.rand(len(df_test_all)) < 0.95
    df_train_propensities = df_test_all[msk]
    df_test = df_test_all[~msk]

    train_matrix, test_matrix = get_numpy(df_train, df_test)

    inverse_propensities_matrix = get_inverse_propensities(df_train_propensities, df_train, train_matrix, **kwargs)


    if is_cv:
        return df_train, df_test, df_train_propensities, train_matrix, test_matrix, inverse_propensities_matrix


    return train_matrix, test_matrix, inverse_propensities_matrix


def print_results(path = 'dirichlet_try_test_error_MAE.txt'):
    with open(path, 'r')as file:
        err_lst = []
        best_err = float('inf')
        for index, line in enumerate(file):
            splitted_line = line.split()
            test_err = float(splitted_line[14])
            if test_err < best_err:
                best_err = test_err
            if (index + 1) % 10 == 0:
                err_lst.append(best_err)
                best_err = float('inf')
            if (index + 1) % 50 == 0:
                break
        print(f'{path.split("/")[0]} test error: ', sum(err_lst) / len(err_lst))

if __name__ == '__main__':
    pass