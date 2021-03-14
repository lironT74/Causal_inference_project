import numpy as np
from reg_numpy_attempt import *
from sklearn.cluster import KMeans
seed = 100



def cluster_users(path="data/yahoo_data"):
    train_matrix, test_matrix, inverse_propensities_matrix = read_yahoo(path)

    # train_matrix = np.expand_dims(train_matrix, axis=0)

    kmeans = KMeans(n_clusters=5, random_state=0).fit(train_matrix)

    print(kmeans.labels_)

    ratings = {clust: [] for clust in [1,2,3,4,5]}

    for user, label in zip(range(train_matrix.shape[0]), kmeans.labels_):

        user_avg_rating = np.mean(train_matrix[user][train_matrix[user] != 0])
        ratings[label + 1].append(user_avg_rating)

    ratings = {clust: np.mean(ratings[clust]) for clust in [1,2,3,4,5]}

    print(ratings)



if __name__ == '__main__':
    np.random.seed(seed)

    cluster_users()