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

def get_inverse_propensities_clustering(df_train_propensities, df_train, train_matrix, mu, num_clusters=5, return_p_y_r=False):
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
    inv_propensities = {}

    for cluster in range(num_clusters):
        beta = {}
        for item in p_o_item_clusters[cluster].keys():
            beta[item] = mu / (p_o_SUMS_clusters[cluster][item] + mu)
        for item in range(num_of_items):
            for r in range(1, 6):
                # paper_prob = p_y_r_o[r] * p_o * (1 / p_y_r[r])
                # item_only_prob = p_y_r_o[r] * p_o_item[item] * (1 / p_y_r[r])

                cluster_only_prob = p_y_r_o_clusters[cluster][r] * p_o * (1 / p_y_r[r])
                item_cluster_prob = p_y_r_o_clusters[cluster][r] * p_o_item_clusters[cluster][item] * (1 / p_y_r[r])

                popularity_cluster_prob = (1-beta[item])*item_cluster_prob + beta[item]*cluster_only_prob
                inv_propensities[(r, item, cluster)] = 1 / popularity_cluster_prob

            inv_propensities[(0, item, cluster)] = 0

    inverse_propensities_matrix = np.zeros((num_of_users, num_of_items))
    for user in range(num_of_users):
        for item in range(num_of_items):
            r = train_matrix[user, item]
            label = clusters_labels[user]
            inverse_propensities_matrix[user, item] = inv_propensities[(r, item + 1, label)]

    if return_p_y_r:
        return inverse_propensities_matrix, p_y_r
    return inverse_propensities_matrix


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
    k_folds = 4

    for i in range(5):

        for mu in [3, 30, 300, 3000, 30000]:
            print(f'START OF ITERATION {i + 1} (MAE)')
            read_data_and_split_to_folds(i + 1,
                                         get_inverse_propensities=get_inverse_propensities_clustering,
                                         delta_type='MAE', path="data/yahoo_data", k=k_folds,
                                         mu=mu, path_to_save_txt=f"MF-IPS mu={mu}/dirichlet_try_mu_{mu}")

            print(f'START OF ITERATION {i + 1} (MSE)')
            read_data_and_split_to_folds(i + 1,
                                         get_inverse_propensities=get_inverse_propensities_clustering,
                                         delta_type='MSE', path="data/yahoo_data", k=k_folds,
                                         mu=mu, path_to_save_txt=f"MF-IPS mu={mu}/dirichlet_try_mu_{mu}")

    for mu in [3, 30, 300, 3000, 30000]:
        print_results(path=f'MF-IPS mu={mu}/dirichlet_try_mu_{mu}_MAE_CV.txt')
        print_results(path=f'MF-IPS mu={mu}/dirichlet_try_mu_{mu}_MSE_CV.txt')

