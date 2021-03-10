import pandas as pd
import pickle
import numpy as np
from scipy.sparse import dok_matrix

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

if __name__ == '__main__':
    read_yahoo()
    # user2song_train, user2song_train_test, user2song_test_test, Y_train, Y_train_test, Y_test_test = load_yahoo()
    #
    # Y_train = Y_train.toarray()
    # Y_train = np.ma.array(Y_train, mask=Y_train <= 0, hard_mask=True, copy=False)
    #
    # Y_train_test = Y_train_test.toarray()
    # Y_train_test = np.ma.array(Y_train_test, mask=Y_train_test <= 0, hard_mask=True, copy=False)
    #
    # prob_obs = Y_train.count() / Y_train.size     # P(O=1)
    # p_y_o_list = {}
    # p_y_r = {}
    #
    # for r in range(1, 6):
    #     p_y_r_given_obs = Y_train[Y_train == r].count() / Y_train.count()    # P(Y=r|O=1)
    #     p_y_o_list[r] = p_y_r_given_obs
    #     p_y_r[r] = Y_train_test[Y_train_test == r].count() / Y_train_test.count()
    #
