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
    p_o_item_popularity = dict(df_train['song_id'].value_counts() / num_of_users)
    p_o_SUMS = dict(df_train['song_id'].value_counts())
    p_o = len(df_train) / train_matrix.size

    alpha = {}
    for item in p_o_item_popularity.keys():
        alpha[item] = mu/(p_o_SUMS[item]+mu)

    propensities = {(r, item): p_y_r_o[r]*((1-alpha[item])*p_o_item_popularity[item]+alpha[item]*p_o)*(1/p_y_r[r]) \
                    for r in p_y_r.keys() \
                    for item in p_o_item_popularity.keys()}

    for item in p_o_item_popularity.keys():
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




def get_MNAR_probs_per_cluster(num_clusters, num_of_items, clusters_matrices):
    p_o_SUMS_clusters = {cluster: {} for cluster in range(num_clusters)}
    p_o_popularity_clusters = {cluster: {} for cluster in range(num_clusters)}

    p_y_r_o_clusters = {cluster: {} for cluster in range(num_clusters)}
    p_o_clusters = {}

    for cluster in range(num_clusters):

        cluster_matrix = clusters_matrices[cluster]
        p_o_clusters[cluster] = cluster_matrix[cluster_matrix != 0].size / cluster_matrix.size

        for rating in range(1, 6):
            p_y_r_o_clusters[cluster][rating] = cluster_matrix[cluster_matrix == rating].size / cluster_matrix[cluster_matrix != 0].size

        for item_index in range(num_of_items):

            counts = len(np.nonzero(cluster_matrix.T[item_index])[0])

            p_o_SUMS_clusters[cluster][item_index] = counts
            p_o_popularity_clusters[cluster][item_index] = counts / cluster_matrix.shape[0]


    return p_o_SUMS_clusters, p_o_popularity_clusters, p_y_r_o_clusters, p_o_clusters


def cluster_popularity_MF_IPS_propensities(df_train_propensities, df_train, train_matrix, return_p_y_r=False,  *args, **kwargs):


    num_clusters = kwargs.get("num_clusters", -1)
    if num_clusters == -1:
        raise ValueError

    use_popularity = kwargs.get("use_popularity", None)
    if use_popularity is None:
        raise ValueError

    mu = kwargs.get("mu", -1)
    if mu == -1 and use_popularity:
        raise ValueError

    num_of_users, num_of_items = train_matrix.shape

    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(train_matrix)

    clusters_labels = kmeans.labels_

    #Clusters estimators:


    #MNAR
    clusters_matrices = {}
    for cluster in range(num_clusters):
        clusters_matrices[cluster] = train_matrix[clusters_labels == cluster]

    p_o_SUMS_clusters, p_o_popularity_clusters, p_y_r_o_clusters, p_o_clusters = get_MNAR_probs_per_cluster(num_clusters,
                                                                                                               num_of_items,
                                                                                                               clusters_matrices)

    # # All observations:
    # p_y_r_o = dict(df_train['rating'].value_counts() / len(df_train))
    # p_o_popularity = dict(df_train['song_id'].value_counts() / num_of_users)
    # p_o_SUMS = dict(df_train['song_id'].value_counts())
    # p_o_all_ = len(df_train) / train_matrix.size
    p_y_r = dict(df_train_propensities['rating'].value_counts() / len(df_train_propensities))
    # alpha = {}
    # for item_index in range(num_of_items):
    #     alpha[item_index] = mu/(p_o_SUMS[item_index] + mu)


    inv_propensities = {}

    for cluster in range(num_clusters):

        beta = {}
        for item_index in range(num_of_items):
            beta[item_index] = mu / (p_o_SUMS_clusters[cluster][item_index] + mu)

        for item_index in range(num_of_items):

            for rating in range(1, 6):

                # FUTURE WORK: interpolate cluster_propensities with this too:
                # if use_popularity:
                #     p_o_all = (1-alpha[item_index])*p_o_popularity[item_index] + alpha[item_index]*p_o_all_
                # else:
                #     p_o_all = p_o_all_
                #
                # all_users_propensities = (p_y_r_o[rating] * p_o_all) * (1 / p_y_r[rating])


                p_o_popularity_cluster_ = p_o_popularity_clusters[cluster][item_index]
                p_o_cluster_ = p_o_clusters[cluster][item_index]

                if use_popularity:
                    p_o_cluster = (1-beta[item_index])*p_o_popularity_cluster_ + beta[item_index]*p_o_cluster_
                else:
                    p_o_cluster = p_o_cluster_

                p_y_r_o_cluster = p_y_r_o_clusters[cluster][rating]

                cluster_propensities = (p_y_r_o_cluster * p_o_cluster) * (1 / p_y_r[rating])

                inv_propensities[(rating, item_index, cluster)] = 1 / cluster_propensities


            inv_propensities[(0, item_index, cluster)] = 0



    inverse_propensities_matrix = np.zeros((num_of_users, num_of_items))
    for user in range(num_of_users):
        for item_index in range(num_of_items):
            rating = train_matrix[user, item_index]
            label = clusters_labels[user]
            inverse_propensities_matrix[user, item_index] = inv_propensities[(rating, item_index + 1, label)]

    if return_p_y_r:
        return inverse_propensities_matrix, p_y_r
    return inverse_propensities_matrix


if __name__ == '__main__':
    df_train, df_test, df_train_propensities, Y, Y_test, inv_propensities = read_yahoo(cluster_popularity_MF_IPS_propensities,
                                                                                       "data/yahoo_data", is_cv=True, num_clusters=3, mu=300)
