import numpy as np
from MF_IPS_numpy_failed import get_numpy
from sklearn.cluster import KMeans
import pandas as pd
seed = 100

def get_probs_per_cluster(num_clusters, num_of_items, clusters):
    p_o_SUMS_clusters = {cluster: {} for cluster in range(num_clusters)}
    p_o_item_clusters = {cluster: {} for cluster in range(num_clusters)}

    p_y_r_o_clusters = {cluster: {} for cluster in range(num_clusters)}
    p_o_clusters = {}

    for cluster in range(num_clusters):
        cluster_matrix = clusters[cluster]
        p_o_clusters[cluster] = cluster_matrix[cluster_matrix != 0].size / cluster_matrix.size
        for rating in range(1, 6):
            p_y_r_o_clusters[cluster][rating] = cluster_matrix[cluster_matrix == rating].size / cluster_matrix[
                cluster_matrix != 0].size
        for item_index in range(num_of_items):
            popularity = len(np.nonzero(cluster_matrix.T[item_index])[0])
            p_o_SUMS_clusters[cluster][item_index] = popularity
            p_o_item_clusters[cluster][item_index] = popularity / cluster_matrix.shape[0]

    return p_o_SUMS_clusters, p_o_item_clusters, p_y_r_o_clusters, p_o_clusters

def get_inverse_propensities_clustering(df_train_propensities, df_train, train_matrix, num_clusters=5, return_p_y_r=False):
    num_of_users, num_of_items = train_matrix.shape

    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(train_matrix)

    clusters_labels = kmeans.labels_
    clusters = {}

    for cluster in range(num_clusters):
        clusters[cluster] = train_matrix[clusters_labels == cluster]

    p_y_r = dict(df_train_propensities['rating'].value_counts() / len(df_train_propensities))
    p_y_r_o = dict(df_train['rating'].value_counts() / len(df_train))
    p_o = len(df_train) / train_matrix.size

    p_o_item = dict(df_train['song_id'].value_counts() / num_of_users)
    p_o_SUMS = dict(df_train['song_id'].value_counts())

    p_o_SUMS_clusters, p_o_item_clusters, p_y_r_o_clusters, p_o_clusters = get_probs_per_cluster(num_clusters,
                                                                                                 num_of_items,
                                                                                                 clusters)
    propensities = {}

    propensities = {}

    for cluster in range(num_clusters):
        for item in range(num_of_items):
            for r in range(1, 6):
                paper_prob = p_y_r_o[r] * p_o * (1 / p_y_r[r])
                item_only_prob = p_y_r_o[r] * p_o_item[item] * (1 / p_y_r[r])
                cluster_only_prob = p_y_r_o_clusters[cluster][r] * p_o * (1 / p_y_r[r])
                item_cluster_prob = p_y_r_o_clusters[cluster][r] * p_o_item_clusters[cluster][item] * (1 / p_y_r[r])

                beta = 1

                popularity_cluster_prob = (1-beta)*item_cluster_prob + beta*cluster_only_prob



    # propensities = {r: p_y_r_o[r] * p_o * (1 / p_y_r[r]) for r in p_y_r.keys()}
    # propensities[0] = 0

    p_f = lambda r: 1 / propensities[r] if propensities[r] != 0 else 0
    p_f_func = np.vectorize(p_f)
    propensities_matrix = p_f_func(train_matrix)

    if return_p_y_r:
        return propensities_matrix, p_y_r
    return propensities_matrix


def read_yahoo_clustering(path="data/yahoo_data", is_cv = False):
    column_names = ['user_id', 'song_id', 'rating']

    df_train = pd.read_csv(path+"/ydata-ymusic-rating-study-v1_0-train.txt", '\t', names=column_names)
    df_test_all = pd.read_csv(path+"/ydata-ymusic-rating-study-v1_0-test.txt", '\t', names=column_names)

    msk = np.random.rand(len(df_test_all)) < 0.95
    df_train_propensities = df_test_all[msk]
    df_test = df_test_all[~msk]

    train_matrix, test_matrix = get_numpy(df_train, df_test)

    inverse_propensities_matrix = get_inverse_propensities_clustering(df_train_propensities, df_train, train_matrix)


    if is_cv:
        return df_train, df_test, df_train_propensities, train_matrix, test_matrix, inverse_propensities_matrix


    return train_matrix, test_matrix, inverse_propensities_matrix


# def cluster_users(path="data/yahoo_data"):
#     train_matrix, test_matrix, inverse_propensities_matrix = read_yahoo(path)
#
#     train_matrix = np.expand_dims(train_matrix, axis=0)
#
#     kmeans = KMeans(n_clusters=2, random_state=0).fit(train_matrix)
#
#     print(kmeans.labels_)


if __name__ == '__main__':
    np.random.seed(seed)
    read_yahoo_clustering(path="data/yahoo_data", is_cv=False)
