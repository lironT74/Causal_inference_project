import numpy as np
from scipy.optimize import minimize
from numpy.linalg import norm
from LoadData import load_yahoo


def params2vec(V, W, a, b, c):
    V_new = V.flatten()
    W_new = W.flatten()
    a_new = a.flatten()
    b_new = b.flatten()
    c_new = c.flatten()
    return np.concatenate([V_new, W_new, a_new, b_new, c_new])


def vec2params(params, num_of_users, num_of_items, inner_dim):
    c = params[-1]
    V = params[:num_of_users*inner_dim]
    W = params[num_of_users*inner_dim:num_of_users*inner_dim+num_of_items*inner_dim]
    remaining_params = params[num_of_users*inner_dim+num_of_items*inner_dim:-1]
    a = remaining_params[:num_of_users]
    b = remaining_params[num_of_users:]
    return V.reshape(num_of_users, inner_dim), W.reshape(num_of_items, inner_dim), a, b, c


def objective_gradient(params_vec, y_observed, propensities,
                       num_of_users, num_of_items, inner_dim, lam, delta_type):
    V, W, a, b, c = vec2params(params_vec, num_of_users, num_of_items, inner_dim)

    scale = num_of_users * num_of_items
    scores = np.matmul(V, W.T) + np.array([a]).T + b + c
    delta = y_observed-scores
    if delta_type == 'MSE':
        delta = delta ** 2
    elif delta_type == 'MAE':
        delta = np.abs(delta)
    objective = (delta * propensities).sum()


    if delta_type == 'MSE':
        gradientMultiplier = propensities * 2 * delta
    elif delta_type == 'MAE':
        gradientMultiplier = np.zeros(delta.shape)
        gradientMultiplier[delta > 0] = 1
        gradientMultiplier[delta < 0] = -1
        gradientMultiplier = propensities * gradientMultiplier
    else:
        raise

    userVGradient = gradientMultiplier @ W
    itemVGradient = gradientMultiplier.T @ V

    userBGradient = gradientMultiplier.sum(1)
    itemBGradient = gradientMultiplier.sum(0)
    globalBGradient = gradientMultiplier.sum()

    scaledPenalty = 1.0 * lam * scale / (num_of_users + num_of_items)
    scaledPenalty /= (inner_dim + 1)

    regularization = lam*(norm(V)**2+norm(W)**2+norm(a)**2+norm(b)**2+c**2)
    objective += regularization

    userVGradient += 2*scaledPenalty*V
    itemVGradient += 2*scaledPenalty*W

    userBGradient += 2*scaledPenalty*a
    itemBGradient += 2*scaledPenalty*b
    globalBGradient += 2*scaledPenalty*c

    gradient = params2vec(userVGradient, itemVGradient, userBGradient, itemBGradient, globalBGradient)
    print(f'Objective: {objective}')
    print(f'Gradient norm: {norm(gradient)}')

    return objective, gradient


def train_mf(y_observed, propensities, delta_type, inner_dim=1000, lam=1):
    num_of_users, num_of_items = y_observed.shape
    inverse_propensities = 1 / propensities

    y_observed = np.ma.filled(y_observed, 0)
    inverse_propensities = np.ma.filled(inverse_propensities, 0)

    V = np.random.randn(num_of_users, inner_dim)
    W = np.random.randn(num_of_items, inner_dim)
    a = np.zeros(num_of_users, dtype=np.longdouble)
    b = np.zeros(num_of_items, dtype=np.longdouble)
    c = np.zeros(1)

    x0 = params2vec(V, W, a, b, c)

    x = minimize(objective_gradient, x0=x0, jac=True, method='L-BFGS-B', args=(y_observed, inverse_propensities,
                                                                               num_of_users, num_of_items, inner_dim, lam, delta_type))

    return x


def get_observed_and_inverse_yahoo():
    user2song_train, user2song_train_test, user2song_test_test, Y_train, Y_train_test, Y_test_test = load_yahoo()

    Y_train = Y_train.toarray()
    Y_train = np.ma.array(Y_train, mask=Y_train <= 0, hard_mask=True, copy=False)

    Y_train_test = Y_train_test.toarray()
    Y_train_test = np.ma.array(Y_train_test, mask=Y_train_test <= 0, hard_mask=True, copy=False)

    p_o = Y_train.count() / Y_train.size  # P(O=1)
    p_y_o_list = {}

    for r in range(1, 6):
        p_y_o = Y_train[Y_train == r].count() / Y_train.count()
        p_y_r = Y_train_test[Y_train_test == r].count() / Y_train_test.count()
        p_y_o_list[r] = p_y_o * p_o / p_y_r

    propensities = Y_train.copy()
    for r in range(1, 6):
        propensities[propensities == r] = p_y_o_list[r]
    return Y_train, Y_test_test, propensities



if __name__ == '__main__':
    Y_train, Y_test_test, propensities = get_observed_and_inverse_yahoo()

    x = train_mf(Y_train, propensities, 'MSE', inner_dim=5, lam=1)

    # import scipy.sparse
    #
    # rows = [2, 1, 4, 3, 0, 4, 3]
    # cols = [0, 2, 1, 1, 0, 0, 0]
    # vals = [1, 2, 3, 4, 5, 4, 5]
    # checkY = scipy.sparse.coo_matrix((vals, (rows, cols)))
    # checkY = checkY.toarray()
    # print(checkY)
    # checkY = np.ma.array(checkY, mask=checkY <= 0, hard_mask=True, copy=False)
    # print(checkY[checkY==5].count()/checkY.count())
    # print(checkY)
    # print(checkY.count())
    # print(checkY.size)