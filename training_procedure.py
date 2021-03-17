from auxiliary import *
from MF_torch import MF
from propensities_estimation import MF_IPS_propensities, popularity_MF_IPS_propensities, cluster_popularity_MF_IPS_propensities


def read_data_and_split_to_folds(iteration, get_inverse_propensities, path_to_save_txt, delta_type=None, path="data/yahoo_data", k=4, *args, **kwargs):

    df_train, df_test, df_train_propensities, Y, Y_test, inv_propensities = read_yahoo(get_inverse_propensities, path, is_cv=True, **kwargs)

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    mse_dict_total = {(lam, inner_dim): 0 for lam in lams for inner_dim in inner_dims}
    mae_dict_total = {(lam, inner_dim): 0 for lam in lams for inner_dim in inner_dims}

    num_users, num_songs = Y.shape

    for fold_num, (train_index, val_index) in enumerate(kf.split(df_train)):

        fold_train_df = df_train.iloc[train_index]
        fold_val_df = df_train.iloc[val_index]

        Y_train = np.zeros((num_users, num_songs))
        Y_val = np.zeros((num_users, num_songs))

        for index, row in fold_train_df.iterrows():
            Y_train[row['user_id'] - 1, row['song_id'] - 1] = row['rating']

        for index, row in fold_val_df.iterrows():
            Y_val[row['user_id'] - 1, row['song_id'] - 1] = row['rating']


        train_propensities = get_inverse_propensities(df_train_propensities, fold_train_df, Y_train, **kwargs)
        val_propensities = get_inverse_propensities(df_train_propensities, fold_val_df, Y_val, **kwargs)

        train_propensities = train_propensities * (k / (k - 1))
        val_propensities = val_propensities * k

        if delta_type is None:

            mse_dict = train_model_CV(Y_train, Y_val, train_propensities, val_propensities, fold_num, iteration,
                                      delta_type="MSE", path_to_save_txt=path_to_save_txt, **kwargs)
            mae_dict = train_model_CV(Y_train, Y_val, train_propensities, val_propensities, fold_num, iteration,
                                      delta_type="MAE", path_to_save_txt=path_to_save_txt, **kwargs)

            for key in mse_dict:
                mse_dict_total[key] += mse_dict[key] / k
                mae_dict_total[key] += mae_dict[key] / k
        else:
            curr_dict = train_model_CV(Y_train, Y_val, train_propensities, val_propensities, fold_num, iteration,
                                      delta_type=delta_type, path_to_save_txt=path_to_save_txt, **kwargs)
            for key in curr_dict:
                mse_dict_total[key] += curr_dict[key] / k

    lam_mse, dim_mse = find_best_key_dict(mse_dict_total)
    lam_mae, dim_mae = find_best_key_dict(mae_dict_total)

    if delta_type is None:
        best_test_err_mse = train_model_test(Y, Y_test, inv_propensities, iteration, "MSE", dim_mse, lam_mse,
                                             path_to_save_txt, **kwargs)
        best_test_err_mae = train_model_test(Y, Y_test, inv_propensities, iteration, "MAE", dim_mae, lam_mae,
                                             path_to_save_txt, **kwargs)

        return best_test_err_mse, best_test_err_mae
    else:
        return train_model_test(Y, Y_test, inv_propensities, iteration,
                                delta_type, dim_mse, lam_mse, path_to_save_txt, **kwargs)


def train_model_CV(Y_train, Y_val, train_propensities, val_propensities, fold_num, iteration, delta_type, path_to_save_txt='torch_find_params', *args, **kwargs):

    use_popularity = kwargs.get("use_popularity", None)
    if use_popularity is None:
        raise ValueError

    mu = kwargs.get("mu", -1)
    if mu == -1 and use_popularity:
        raise ValueError

    num_clusters = kwargs.get("num_clusters", -1)
    if num_clusters == -1:
        raise ValueError


    num_users, num_items = Y_train.shape
    Y_train = torch.from_numpy(Y_train)
    Y_val = torch.from_numpy(Y_val)
    train_propensities = torch.from_numpy(train_propensities)
    val_propensities = torch.from_numpy(val_propensities)

    val_err_dict = {}
    print()
    with open(f'{path_to_save_txt}_{delta_type}_CV.txt', 'a') as f:
        for inner_dim in inner_dims:
            for lam in lams:
                model = MF(num_users, num_items, inner_dim, Y_train, Y_val, train_propensities, delta_type, lam)

                optimizer = torch.optim.LBFGS(model.parameters())
                val_err_dict[(lam, inner_dim)] = float('inf')
                for epoch in range(EPOCHS):
                    optimizer.zero_grad()
                    loss = model()
                    with torch.no_grad():
                        train_err, val_err = model.calc_train_val_err_ips(train_propensities, val_propensities)
                        if val_err < val_err_dict[(lam, inner_dim)]:
                            val_err_dict[(lam, inner_dim)] = val_err
                        output_txt = f'iteration: {iteration} \t delta_type: {delta_type} \t fold: {fold_num + 1} \t epoch: {epoch + 1}. loss: {loss} \t train err: {train_err} \t val err: {val_err} \t lam: {lam} \t inner_dim: {inner_dim} '
                        print(output_txt)
                        f.write(f'{output_txt}\n')

                    def closure():
                        optimizer.zero_grad()
                        loss = model()
                        loss.backward()
                        return loss

                    optimizer.step(closure)

    return val_err_dict


def train_model_test(Y, Y_test, inv_propensities, iteration, delta_type, best_dim, best_lam, path_to_save_txt='test_error', *args, **kwargs):


    use_popularity = kwargs.get("use_popularity", None)
    if use_popularity is None:
        raise ValueError

    mu = kwargs.get("mu", -1)
    if mu == -1 and use_popularity:
        raise ValueError

    cluster_size = kwargs.get('cluster_size', -1)
    if cluster_size == -1:
        raise ValueError

    num_users, num_items = Y.shape
    Y = torch.from_numpy(Y)
    Y_test = torch.from_numpy(Y_test)
    train_propensities = torch.from_numpy(inv_propensities)
    inner_dim = best_dim
    lam = best_lam
    best_test_err = float('inf')
    print()
    print()
    with open(f'{path_to_save_txt}_{delta_type}.txt', 'a') as f:
        model = MF(num_users, num_items, inner_dim, Y, Y_test, train_propensities, delta_type, lam)

        optimizer = torch.optim.LBFGS(model.parameters())

        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            loss = model()

            with torch.no_grad():
                train_err, test_err = model.calc_train_test_err()
                output_txt = f'iteration: {iteration} \t k: {cluster_size} \t delta type: {delta_type}\t epoch: {epoch + 1}. loss: {loss} \t train err: {train_err} \t test err: {test_err} \t lam: {lam} \t inner_dim: {inner_dim} '
                print(output_txt)
                f.write(f'{output_txt}\n')
                if best_test_err > test_err:
                    best_test_err = test_err

            def closure():
                optimizer.zero_grad()
                loss = model()
                loss.backward()
                return loss

            optimizer.step(closure)

    with open(f'{path_to_save_txt}_{delta_type}_best.txt', 'a') as f:
        output_txt = f'iteration: {iteration}\t k: {cluster_size} \ttest err: {best_test_err} \t lam: {lam} \t inner_dim: {inner_dim} '
        print(f'delta type: {delta_type}' + " " + output_txt)
        f.write(f'{output_txt}\n')

    return best_test_err



if __name__ == '__main__':
    k_folds = 4
    delta_type = 'MSE'
    num_clusterssss = [3, 5, 10]
    for i in range(5):
        for num_clusters in num_clusterssss:
            print(f'START OF ITERATION {i + 1}')
            print(f'num_clusters: {num_clusters}, delta: {delta_type}')

            dir = f'cluster-MF-IPS/num_clusters={num_clusters}/'

            os.makedirs(dir, exist_ok=True)
            read_data_and_split_to_folds(iteration=i + 1,
                                         get_inverse_propensities=cluster_popularity_MF_IPS_propensities,
                                         path_to_save_txt=f"{dir}exp",
                                         delta_type=delta_type,
                                         path="data/yahoo_data",
                                         k=k_folds,
                                         use_popularity=False,
                                         num_clusters=num_clusters,
                                         mu=5)

    for num_clusters in num_clusterssss:
        print(f'num_clusters: {num_clusters}')
        dir = f'cluster-MF-IPS/num_clusters={num_clusters}/'
        print_results(path=f'{dir}exp_{delta_type}_best.txt', at_index=6, epochs=7)


