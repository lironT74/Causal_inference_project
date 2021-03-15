from auxiliary import *

def MF_IPS_propensities(df_train_propensities, df_train, train_matrix, return_p_y_r=False, *args, **kwargs):

    p_y_r = dict(df_train_propensities['rating'].value_counts() / len(df_train_propensities))
    p_y_r_o = dict(df_train['rating'].value_counts() / len(df_train))
    p_o = len(df_train) / train_matrix.size

    propensities = {r: p_y_r_o[r]*p_o*(1/p_y_r[r]) for r in p_y_r.keys()}
    propensities[0] = 0

    p_f = lambda r: 1/propensities[r] if propensities[r] != 0 else 0
    p_f_func = np.vectorize(p_f)
    propensities_matrix = p_f_func(train_matrix)

    if return_p_y_r:
        return propensities_matrix, p_y_r
    return propensities_matrix


def popularity_MF_IPS_propensities(df_train_propensities, df_train, train_matrix, return_p_y_r=False, *args, **kwargs):

    mu = kwargs.get("mu", -1)
    if mu == -1:
        raise ValueError


    num_of_users, num_of_items = train_matrix.shape

    p_y_r = dict(df_train_propensities['rating'].value_counts() / len(df_train_propensities))
    p_y_r_o = dict(df_train['rating'].value_counts() / len(df_train))
    p_o_item = dict(df_train['song_id'].value_counts() / num_of_users)
    p_o_SUMS = dict(df_train['song_id'].value_counts())
    p_o = len(df_train) / train_matrix.size

    alpha = {}
    for item in p_o_item.keys():
        alpha[item] = mu/(p_o_SUMS[item]+mu)

    propensities = {(r, item): p_y_r_o[r]*((1-alpha[item])*p_o_item[item]+alpha[item]*p_o)*(1/p_y_r[r]) \
                    for r in p_y_r.keys() \
                    for item in p_o_item.keys()}

    for item in p_o_item.keys():
        propensities[(0, item)] = 0

    p_f = lambda r, item: 1/propensities[(r, item)] if propensities[(r,item)] != 0 else 0

    inverse_propensities_matrix = np.zeros((num_of_users, num_of_items))
    for user in range(num_of_users):
        for item in range(num_of_items):
            r = train_matrix[user, item]
            inverse_propensities_matrix[user, item] = p_f(r, item+1)

    if return_p_y_r:
        return inverse_propensities_matrix, p_y_r
    return inverse_propensities_matrix


def get_probs_per_cluster(num_clusters, num_of_items, clusters):
    p_o_SUMS_clusters = {cluster: {} for cluster in range(num_clusters)}
    p_o_item_clusters = {cluster: {} for cluster in range(num_clusters)}

    p_y_r_o_clusters = {cluster: {} for cluster in range(num_clusters)}
    p_o_clusters = {}

    for cluster in range(num_clusters):
        cluster_matrix = clusters[cluster]
        p_o_clusters[cluster] = cluster_matrix[cluster_matrix != 0].size / cluster_matrix.size
        for rating in range(1, 6):
            p_y_r_o_clusters[cluster][rating] = cluster_matrix[cluster_matrix == rating].size / cluster_matrix[cluster_matrix != 0].size

        for item_index in range(num_of_items):
            popularity = len(np.nonzero(cluster_matrix.T[item_index])[0])
            p_o_SUMS_clusters[cluster][item_index] = popularity
            p_o_item_clusters[cluster][item_index] = popularity / cluster_matrix.shape[0]

    return p_o_SUMS_clusters, p_o_item_clusters, p_y_r_o_clusters, p_o_clusters


def cluster_popularity_MF_IPS_propensities(df_train_propensities, df_train, train_matrix, return_p_y_r=False,  *args, **kwargs):

    mu = kwargs.get("mu", -1)
    if mu == -1:
        raise ValueError

    num_clusters = kwargs.get("num_clusters", -1)
    if num_clusters == -1:
        raise ValueError

    num_of_users, num_of_items = train_matrix.shape

    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(train_matrix)

    clusters_labels = kmeans.labels_
    clusters = {}

    for cluster in range(num_clusters):
        clusters[cluster] = train_matrix[clusters_labels == cluster]

    p_y_r = dict(df_train_propensities['rating'].value_counts() / len(df_train_propensities))
    # p_y_r_o = dict(df_train['rating'].value_counts() / len(df_train))
    p_o = len(df_train) / train_matrix.size

    # p_o_item = dict(df_train['song_id'].value_counts() / num_of_users)

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


if __name__ == '__main__':
    pass