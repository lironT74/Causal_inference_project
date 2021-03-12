import numpy as np
import scipy
import pandas as pd
import tqdm
from scipy.optimize import minimize


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


def get_inverse_propensities(df_train_propensities, df_train, train_matrix):

    p_y_r = dict(df_train_propensities['rating'].value_counts() / len(df_train_propensities))
    p_y_r_o = dict(df_train['rating'].value_counts() / len(df_train))
    p_o = len(df_train) / train_matrix.size

    propensities = {r: p_y_r_o[r]*p_o*(1/p_y_r[r]) for r in p_y_r.keys()}
    propensities[0] = 0

    p_f = lambda r: 1/propensities[r] if propensities[r] != 0 else 0
    p_f_func = np.vectorize(p_f)
    propensities_matrix = p_f_func(train_matrix)

    return propensities_matrix


def read_yahoo(path="data/yahoo_data"):
    column_names = ['user_id', 'song_id', 'rating']

    df_train = pd.read_csv(path+"/ydata-ymusic-rating-study-v1_0-train.txt", '\t', names=column_names)
    df_test_all = pd.read_csv(path+"/ydata-ymusic-rating-study-v1_0-test.txt", '\t', names=column_names)

    msk = np.random.rand(len(df_test_all)) < 0.95
    df_train_propensities = df_test_all[msk]
    df_test = df_test_all[~msk]

    train_matrix, test_matrix = get_numpy(df_train, df_test)
    inverse_propensities_matrix = get_inverse_propensities(df_train_propensities, df_train, train_matrix)

    return train_matrix, test_matrix, inverse_propensities_matrix


def get_problem_parameters(inner_dim, shape):

    num_users, num_songs = shape

    V = np.random.rand(num_users, inner_dim)
    W = np.random.rand(num_songs, inner_dim)

    # A = np.random.rand(num_users, 1)
    # B = np.random.rand(1, num_songs)
    # C = np.random.rand(1)

    A = np.zeros((num_users))
    B = np.zeros((num_songs))
    C = np.zeros(1)

    return np.concatenate([V.flatten(), W.flatten(), A.flatten(), B.flatten(), C.flatten()])


def get_params_from_vec(vec, inner_dim, shape):
    num_users, num_songs = shape

    end_V_W = inner_dim * (num_users + num_songs)

    V = vec[:num_users*inner_dim]
    V = V.reshape(num_users, inner_dim)

    W = vec[num_users*inner_dim: end_V_W]
    W = W.reshape(num_songs, inner_dim)

    A = vec[end_V_W: end_V_W + num_users]

    B = vec[end_V_W + num_users: end_V_W + num_users + num_songs]

    C = vec[-1]

    return V, W, A, B, C


def get_y_hat(V, W, A, B, C):
    return V @ W.T + np.expand_dims(A, axis=0).T + B + C


def get_objective(vec, inverse_propensities_matrix, Y, Y_test, inner_dim, shape, lam, type_loss):
    V, W, A, B, C = get_params_from_vec(vec, inner_dim, shape)

    Y_hat = get_y_hat(V, W, A, B, C)

    difference = np.abs(Y - Y_hat)

    if type == 'MSE':
        difference = np.square(difference)

    sum_delta_propensities = np.sum(difference * inverse_propensities_matrix)

    regularization = lam * (np.square(np.linalg.norm(V, 'fro')) +
                            np.square(np.linalg.norm(W, 'fro')) +
                            np.square(np.linalg.norm(A)) +
                            np.square(np.linalg.norm(B)) +
                            np.square(C))


    print(f"test error is: {get_error(vec, Y_test, type_loss)} {lam, inner_dim}")

    return sum_delta_propensities + regularization


def get_gradient(vec, inverse_propensities_matrix, Y, Y_test, inner_dim, shape, lam, type_loss):

    V, W, A, B, C = get_params_from_vec(vec, inner_dim, shape)

    Y_hat = get_y_hat(V, W, A, B, C)

    diff = 2 * (Y - Y_hat)

    if type_loss == "MAE":
        # sub gradient:
        diff[diff > 0] = 1
        diff[diff < 0] = - 1

    diff = inverse_propensities_matrix * diff

    V_grad = diff @ W + lam * V * 2
    W_grad = diff.T @ V + lam * W * 2
    A_grad = np.sum(diff, axis = 1) + lam * A * 2
    B_grad = np.sum(diff, axis = 0) + lam * B * 2
    C_grad = np.sum(diff) + lam * C * 2

    return np.concatenate([V_grad.flatten(), W_grad.flatten(), A_grad.flatten(), B_grad.flatten(), C_grad.flatten()])


def get_error(vec, Y, type_loss='MSE'):
    V, W, A, B, C = get_params_from_vec(vec, inner_dim, shape)

    Y_hat = get_y_hat(V, W, A, B, C)

    Y_hat[Y == 0] = 0

    if type_loss == 'MSE':
        return np.sum((np.square(Y - Y_hat))) / len(Y[Y != 0])
    else:
        return np.sum((np.abs(Y - Y_hat))) / len(Y[Y != 0])


if __name__ == '__main__':
    inner_dim = 10

    Y, Y_test, inverse_propensities_matrix = read_yahoo(path="data/yahoo_data")
    shape = Y.shape
    vec = get_problem_parameters(inner_dim, shape)

    lam = 1.0
    type_loss = 'MSE'


    for lam in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]:
        for inner_dim in [5, 10, 20, 40]:
            x = minimize(get_objective, x0=vec, jac=get_gradient, method='L-BFGS-B', options={'maxiter': 20},
                         args=(inverse_propensities_matrix, Y, Y_test, inner_dim, shape, lam, type_loss))

