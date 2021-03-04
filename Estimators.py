import numpy as np

def r_naive(delta, Y_hat, Obs):
    return sum(delta(y_i, o_i) for y_i, o_i in zip(Y_hat, Obs)) / len(Obs)

def ips(delta, y_hat, o, probs):
    return sum(delta(y_i, o_i) for y_i, o_i in zip(Y_hat, Obs)) / len(Obs)
