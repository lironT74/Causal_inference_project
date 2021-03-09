import numpy as np
from scipy.optimize import minimize
from numpy.linalg import norm

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

    # y_observed = np.ma.filled(y_observed, 0) # Make an assumption, that it is done outside
    # propensities = np.ma.filled(propensities, 0) # The same
    scores = np.matmul(V, W.T) + a + b + c
    delta = y_observed-scores
    if delta_type == 'MSE':
        delta = delta ** 2
    elif delta_type == 'MAE':
        delta = np.abs(delta)
    objective = (delta * propensities).sum()
    regularization = lam*(norm(V)**2+norm(W)**2+norm(a)**2+norm(b)**2+c**2)
    objective += regularization

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

    userVGradient += 2*lam*V
    itemVGradient += 2*lam*W

    userBGradient += 2*lam*a
    itemBGradient += 2*lam*b
    globalBGradient += 2*lam*c

    gradient = params2vec(userVGradient, itemVGradient, userBGradient, itemBGradient, globalBGradient)

    return objective, gradient

def calculate_propencities(observed_ratings, inverse_propensities):
    pass

if __name__ == '__main__':
    import scipy.sparse

    rows = [2, 1, 4, 3, 0, 4, 3]
    cols = [0, 2, 1, 1, 0, 0, 0]
    vals = [1, 2, 3, 4, 5, 4, 5]
    checkY = scipy.sparse.coo_matrix((vals, (rows, cols)))
    checkY = checkY.toarray()
    print(checkY)
    checkY = np.ma.array(checkY, mask=checkY <= 0, hard_mask=True, copy=False)
    print(checkY[checkY==5].count()/checkY.count())
    print(checkY.size)
    # randomPropensities = np.random.rand(Y_train.shape[0], Y_train.shape[1])
    # print(randomPropensities)
    # randomInvPropensities = np.reciprocal(randomPropensities)
    # print(randomInvPropensities)