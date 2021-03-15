from scipy.optimize import minimize
from auxiliary import *


def get_problem_parameters(inner_dim, shape):

    num_users, num_songs = shape

    V = np.random.randn(num_users, inner_dim)
    W = np.random.randn(num_songs, inner_dim)

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


    print(f"test error is: {get_error(vec, Y_test, shape, inverse_propensities_matrix, inner_dim, type_loss)} {lam, inner_dim}")

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


def get_error(vec, Y, shape, inverse_propensities_matrix, inner_dim, type_loss='MSE'):
    V, W, A, B, C = get_params_from_vec(vec, inner_dim, shape)

    Y_hat = get_y_hat(V, W, A, B, C)

    # Y_hat[Y == 0] = 0
    # if type_loss == 'MSE':
    #     return np.sum((np.square(Y - Y_hat))) / len(Y[Y != 0])
    # else:
    #     return np.sum((np.abs(Y - Y_hat))) / len(Y[Y != 0])

    delta = Y_hat - Y
    if type_loss == 'MSE':
        delta = np.square(delta)
    elif type_loss == 'MAE':
        delta = np.ma.abs(delta)

    numUsers, numItems = shape
    scale = numUsers * numItems

    observedError = delta * inverse_propensities_matrix
    cumulativeError = np.sum(observedError)
    vanillaMetric = cumulativeError / scale

    return vanillaMetric


if __name__ == '__main__':

    Y, Y_test, inverse_propensities_matrix = read_yahoo(path="data/yahoo_data")
    shape = Y.shape

    type_loss = 'MSE'

    vec = get_problem_parameters(20, shape)
    x = minimize(get_objective, x0=vec, jac=get_gradient, method='L-BFGS-B', options={'maxiter': 20},
                 args=(inverse_propensities_matrix, Y, Y_test, 20, shape, 1, type_loss))