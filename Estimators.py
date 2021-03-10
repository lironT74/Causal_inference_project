import numpy as np
from scipy.optimize import minimize
from numpy.linalg import norm
from LoadData import load_yahoo
from collections import OrderedDict


def params2vec(V, W, a, b, c):
    V_new = V.flatten()
    W_new = W.flatten()
    a_new = a.flatten()
    b_new = b.flatten()
    c_new = c.flatten()
    # print('Params:')
    # print(np.concatenate([V_new, W_new, a_new, b_new, c_new]))
    return np.concatenate([V_new, W_new, a_new, b_new, c_new])


def vec2params(params, num_of_users, num_of_items, inner_dim):
    c = params[-1]
    V = params[:num_of_users*inner_dim]
    W = params[num_of_users*inner_dim:num_of_users*inner_dim+num_of_items*inner_dim]
    remaining_params = params[num_of_users*inner_dim+num_of_items*inner_dim:-1]
    a = remaining_params[:num_of_users]
    b = remaining_params[num_of_users:]
    return V.reshape(num_of_users, inner_dim), W.reshape(num_of_items, inner_dim), a, b, c


def objective_gradient(params_vec, y_observed, y_test, propensities,
                       num_of_users, num_of_items, inner_dim, lam, delta_type):
    V, W, a, b, c = vec2params(params_vec, num_of_users, num_of_items, inner_dim)

    scale = num_of_users * num_of_items
    scores = np.matmul(V, W.T) + np.array([a]).T + b + c
    delta = y_observed-scores

    train_error = y_observed[y_observed != 0]-scores[y_observed != 0]
    # print(y_test[y_test != 0].shape)
    test_error = y_test[y_test != 0] - scores[y_test != 0]
    if delta_type == 'MSE':
        delta = delta ** 2
        train_error = train_error ** 2
        test_error = test_error ** 2
    elif delta_type == 'MAE':
        delta = np.abs(delta)
        train_error = np.abs(train_error)
        test_error = np.abs(test_error)


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

    regularization = lam*(norm(V, 'fro')**2+norm(W, 'fro')**2+norm(a)**2+norm(b)**2+c**2)
    objective += regularization

    userVGradient += 2*scaledPenalty*V
    itemVGradient += 2*scaledPenalty*W

    userBGradient += 2*scaledPenalty*a
    itemBGradient += 2*scaledPenalty*b
    globalBGradient += 2*scaledPenalty*c

    gradient = params2vec(userVGradient, itemVGradient, userBGradient, itemBGradient, globalBGradient)

    test_error_val = test_error.sum()/y_test[y_test!=0].size

    print()
    print(f'Objective: {objective}')
    print(f'Gradient norm: {norm(gradient)}')
    print(f'Train error: {train_error.sum()/y_observed[y_observed!=0].size}')
    print(f'Test error: {test_error_val}')


    with open(f'param_search_{delta_type}.txt', 'a') as f:
        f.write(f"{lam}_{inner_dim}_{test_error_val}\n")

    return objective, gradient


def train_mf(y_observed, y_test, propensities, delta_type, inner_dim=1000, lam=1):
    num_of_users, num_of_items = y_observed.shape
    inverse_propensities = 1 / propensities

    y_observed = np.ma.filled(y_observed, 0)
    inverse_propensities = np.ma.filled(inverse_propensities, 0)

    y_test = np.ma.filled(y_test, 0)

    # print(inverse_propensities.size)
    # print(inverse_propensities[inverse_propensities != 0].size)
    # print(y_observed[y_observed != 0].size)
    # print(np.argwhere(y_observed==0))

    V = np.random.rand(num_of_users, inner_dim)
    W = np.random.rand(num_of_items, inner_dim)
    a = np.zeros(num_of_users, dtype=np.longdouble)
    b = np.zeros(num_of_items, dtype=np.longdouble)
    c = np.zeros(1)

    x0 = params2vec(V, W, a, b, c)
    # print('blu')
    # print(x0)
    # new_V, new_W, new_a, new_b, new_c = vec2params(x0, num_of_users, num_of_items, inner_dim)
    x = minimize(objective_gradient, x0=x0, jac=True, method='L-BFGS-B', options = {'maxiter': 20},
                 args=(y_observed, y_test, inverse_propensities, num_of_users, num_of_items, inner_dim, lam, delta_type)
    )

    return x


def get_observed_and_inverse_yahoo():
    user2song_train, user2song_train_test, user2song_test_test, Y_train, Y_train_test, Y_test_test = load_yahoo()

    Y_train = Y_train.toarray()
    Y_train = np.ma.array(Y_train, mask=Y_train <= 0, hard_mask=True, copy=False)

    Y_train_test = Y_train_test.toarray()
    Y_train_test = np.ma.array(Y_train_test, mask=Y_train_test <= 0, hard_mask=True, copy=False)
    # print('Y train test count', Y_train_test.count())

    Y_test_test = Y_test_test.toarray()
    Y_test_test = np.ma.array(Y_test_test, mask=Y_test_test <= 0, hard_mask=True, copy=False)
    # print('Y test test count', Y_test_test.count())
    # print(Y_test_test)

    p_o = Y_train.count() / Y_train.size  # P(O=1)
    p_y_o_dict = {}

    for r in range(1, 6):
        p_y_o = Y_train[Y_train == r].count() / Y_train.count()
        p_y_r = Y_train_test[Y_train_test == r].count() / Y_train_test.count()
        p_y_o_dict[r] = p_y_o * p_o / p_y_r

    propensities = Y_train.copy()
    for r in range(1, 6):
        propensities[propensities == r] = p_y_o_dict[r]
    return Y_train, Y_test_test, propensities

def calc_test_score(params, y_test, delta_type):
    V, W, a, b, c = params

    scores = np.matmul(V, W.T) + np.array([a]).T + b + c
    delta = y_test - scores

    if delta_type == 'MSE':
        delta = delta ** 2
    elif delta_type == 'MAE':
        delta = np.abs(delta)

    return delta.mean()



def lowest_test_error():

    acc_dict = OrderedDict()
    with open('param_search_MSE.txt', 'r') as f:
        lines = f.read().splitlines()

        for line in lines:
            lam, d, acc = line.split('_')

            if (lam, d) not in acc_dict:
                acc_dict[(lam, d)] = acc

            elif acc_dict[(lam, d)] < acc:
                acc_dict[(lam, d)] = acc


    for (lam, d), acc in acc_dict.items():
        print(f"lambda={lam}, inner dim = {d}, test acc = {acc}")


if __name__ == '__main__':
    Y_train, Y_test_test, propensities = get_observed_and_inverse_yahoo()

    possible_d = [5, 10, 20, 40]
    possible_lam = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]

    # best_error = np.inf
    # best_params = (possible_lam[0], possible_d[0])

    for d in possible_d:
        for lam in possible_lam:
            print(f"lambda: {lam}, inner dimension: {d}:")
            train_mf(Y_train, Y_test_test, propensities, 'MSE', inner_dim=d, lam=lam)

    lowest_test_error()


    # res = train_mf(Y_train, Y_test_test, propensities, 'MSE', inner_dim=inner_dim, lam=1)


    # x = res.x
    # num_of_users, num_of_items = Y_train.shape
    # V, W, a, b, c = vec2params(x, num_of_users, num_of_items, inner_dim)


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
    #
    # import scipy.sparse
    #
    # rows = [2, 1, 4, 3, 0, 4, 3]
    # cols = [0, 2, 1, 1, 0, 0, 0]
    # vals = [1, 2, 3, 4, 5, 4, 5]
    # checkY = scipy.sparse.coo_matrix((vals, (rows, cols)))
    # checkY = checkY.toarray()
    # checkY = np.ma.array(checkY, mask=checkY <= 0, hard_mask=True, copy=False)
    # print("[MAIN]\t Partially observed ratings matrix")
    # print(checkY)
    # randomPropensities = np.random.rand(*np.shape(checkY))
    # x = train_mf(checkY, randomPropensities, 'MSE', inner_dim=3, lam=1)