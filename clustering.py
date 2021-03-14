import numpy as np
from reg_numpy_attempt import *
from sklearn.cluster import KMeans
seed = 100



def cluster_users(path="data/yahoo_data"):
    train_matrix, test_matrix, inverse_propensities_matrix = read_yahoo(path)

    train_matrix = np.expand_dims(train_matrix, axis=0)

    kmeans = KMeans(n_clusters=2, random_state=0).fit(train_matrix)

    print(kmeans.labels_)


if __name__ == '__main__':
    np.random.seed(seed)

    cluster_users()