import torch.nn as nn
import torch
import numpy as np
from reg_numpy_attempt import read_yahoo, get_inverse_propensities
from sklearn.model_selection import KFold

seed = 100

class MF(nn.Module):
    def __init__(self, num_users, num_items, inner_dim, Y, Y_test, inv_propensities, delta_type, lam):
        super(MF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.inner_dim = inner_dim
        self.V = nn.Parameter(torch.rand((num_users, inner_dim)))
        self.W = nn.Parameter(torch.rand((num_items, inner_dim)))
        self.a = nn.Parameter(torch.zeros(num_users))
        self.b = nn.Parameter(torch.zeros(num_items))
        self.c = nn.Parameter(torch.zeros(1))
        self.inv_propensities = inv_propensities
        self.Y = Y
        self.Y_test = Y_test
        self.delta_type = delta_type
        self.lam = lam
        self.scale = self.num_users * self.num_items

    def forward(self):
        scaledPenalty = 1.0 * self.lam * self.scale / (self.num_users + self.num_items)
        scaledPenalty /= (self.inner_dim + 1)

        if self.delta_type == "MSE":
            scores = self.V @ self.W.T+self.a.unsqueeze(-1)+self.b+self.c
            diff = (self.Y - scores)**2
        else:
            scores = self.V @ self.W.T + self.a.unsqueeze(-1) + self.b + self.c
            diff = torch.abs(self.Y - scores)

        regularization = torch.norm(self.V)**2 + torch.norm(self.W)**2 + torch.norm(self.a)**2
        regularization += torch.norm(self.b)**2 + torch.norm(self.c)**2

        diff = self.inv_propensities * diff

        objective = diff.sum() + scaledPenalty * regularization


        return objective

    def calc_train_test_err(self):
        scores = self.V @ self.W.T+self.a.unsqueeze(-1)+self.b+self.c

        if self.delta_type == "MSE":
            diff_train = (self.Y[self.Y != 0] - scores[self.Y != 0])**2
            diff_test = (self.Y_test[self.Y_test != 0] - scores[self.Y_test != 0]) ** 2
        else:
            diff_train = torch.abs(self.Y[self.Y != 0] - scores[self.Y != 0])
            diff_test = torch.abs(self.Y_test[self.Y_test != 0] - scores[self.Y_test != 0])

        return diff_train.mean(), diff_test.mean()

    def calc_train_val_err_ips(self, train_propensities, val_propensities):
        scores = self.V @ self.W.T+self.a.unsqueeze(-1)+self.b+self.c

        if self.delta_type == "MSE":
            diff_train = (self.Y - scores)**2
            diff_test = (self.Y_test - scores) ** 2
        else:
            diff_train = torch.abs(self.Y - scores)
            diff_test = torch.abs(self.Y_test - scores)

        return (train_propensities*diff_train).mean(), (val_propensities*diff_test).mean()


def read_data_and_split_to_folds(iteration, delta_type=None, path="data/yahoo_data", k=4):
    df_train, df_test, df_train_propensities, Y, Y_test, inv_propensities = read_yahoo(path, is_cv=True)

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    inner_dims = [5]
    lams = [1e-4, 1e-3, 1e-2, 1e-1, 1]

    mse_dict_total = {(lam, inner_dim): 0 for lam in lams for inner_dim in inner_dims}
    mae_dict_total = {(lam, inner_dim): 0 for lam in lams for inner_dim in inner_dims}

    num_users = 15400
    num_songs = 1000

    for fold_num, (train_index, val_index) in enumerate(kf.split(df_train)):

        fold_train_df = df_train.iloc[train_index]
        fold_val_df = df_train.iloc[val_index]

        Y_train = np.zeros((num_users, num_songs))
        Y_val = np.zeros((num_users, num_songs))

        for index, row in fold_train_df.iterrows():
            Y_train[row['user_id'] - 1, row['song_id'] - 1] = row['rating']

        for index, row in fold_val_df.iterrows():
            Y_val[row['user_id'] - 1, row['song_id'] - 1] = row['rating']


        train_propensities = get_inverse_propensities(df_train_propensities, fold_train_df, Y_train)
        val_propensities = get_inverse_propensities(df_train_propensities, fold_val_df, Y_val)

        train_propensities = train_propensities * (k / (k - 1))
        val_propensities = val_propensities * k

        if delta_type is None:

            mse_dict = train_model_CV(Y_train, Y_val, train_propensities, val_propensities, fold_num, iteration,
                                      delta_type="MSE")
            mae_dict = train_model_CV(Y_train, Y_val, train_propensities, val_propensities, fold_num, iteration,
                                      delta_type="MAE")

            for key in mse_dict:
                mse_dict_total[key] += mse_dict[key] / k
                mae_dict_total[key] += mae_dict[key] / k
        else:
            curr_dict = train_model_CV(Y_train, Y_val, train_propensities, val_propensities, fold_num, iteration,
                                      delta_type=delta_type)
            for key in curr_dict:
                mse_dict_total[key] += curr_dict[key] / k



    lam_mse, dim_mse = find_best_key_dict(mse_dict_total)
    lam_mae, dim_mae = find_best_key_dict(mae_dict_total)

    if delta_type is None:
        best_test_err_mse = train_model_test(Y, Y_test, inv_propensities, iteration, "MSE", dim_mse, lam_mse)
        best_test_err_mae = train_model_test(Y, Y_test, inv_propensities, iteration, "MAE", dim_mae, lam_mae)
    else:
        return train_model_test(Y, Y_test, inv_propensities, iteration, delta_type, dim_mse, lam_mse)

    return best_test_err_mse, best_test_err_mae


def find_best_key_dict(dict_total):
    return min(dict_total, key=dict_total.get)

def train_model_CV(Y_train, Y_val, train_propensities, val_propensities, fold_num, iteration, delta_type):
    EPOCHS = 10
    num_users, num_items = Y_train.shape
    inner_dims = [5]
    lams = [1e-4, 1e-3, 1e-2, 1e-1, 1]
    Y_train = torch.from_numpy(Y_train)
    Y_val = torch.from_numpy(Y_val)
    train_propensities = torch.from_numpy(train_propensities)
    val_propensities = torch.from_numpy(val_propensities)

    val_err_dict = {}
    print()
    with open(f'torch_find_params_{delta_type}_CV.txt', 'a') as f:
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

def train_model_test(Y, Y_test, inv_propensities, iteration, delta_type, best_dim, best_lam):
    EPOCHS = 10
    num_users, num_items = Y.shape
    Y = torch.from_numpy(Y)
    Y_test = torch.from_numpy(Y_test)
    train_propensities = torch.from_numpy(inv_propensities)
    inner_dim = best_dim
    lam = best_lam
    best_test_err = float('inf')
    print()
    print()
    with open(f'test_error_{delta_type}.txt', 'a') as f:
        model = MF(num_users, num_items, inner_dim, Y, Y_test, train_propensities, delta_type, lam)

        optimizer = torch.optim.LBFGS(model.parameters())

        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            loss = model()

            with torch.no_grad():
                train_err, test_err = model.calc_train_test_err()
                output_txt = f'iteration: {iteration} \t delta type: {delta_type}\t epoch: {epoch + 1}. loss: {loss} \t train err: {train_err} \t test err: {test_err} \t lam: {lam} \t inner_dim: {inner_dim} '
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
    with open(f'best_test_error_{delta_type}.txt', 'a') as f:
        output_txt = f'iteration: {iteration} \t \ttest err: {best_test_err} \t lam: {lam} \t inner_dim: {inner_dim} '
        print(f'delta type: {delta_type}' + " " + output_txt)
        f.write(f'{output_txt}\n')

    return best_test_err


if __name__ == '__main__':
    np.random.seed(seed)
    torch.manual_seed(seed)
    err_list = []
    for i in range(30):
        print(f'START OF ITERATION {i+1}')
        # test_err_mse, test_err_mae = read_data_and_split_to_folds(i+1, path="data/yahoo_data", k=4)
        # mse_err_list.append(test_err_mse)
        # mae_err_list.append(test_err_mae)
        test_err = read_data_and_split_to_folds(i+1, delta_type='MAE', path="data/yahoo_data", k=4)


    # Y, Y_test, inv_propensities = read_yahoo(path="data/yahoo_data")
    # EPOCHS = 10
    # num_users, num_items = Y.shape
    # inner_dims = [5, 10, 20, 40]
    # lams = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    # delta_type = 'MAE'
    # Y = torch.from_numpy(Y)
    # Y_test = torch.from_numpy(Y_test)
    # inv_propensities = torch.from_numpy(inv_propensities)
    # lam = 1.
    # best_test_err = float('inf')
    # best_lam = lams[0]
    # best_dim = inner_dims[0]
    # with open('torch_find_params_mae.txt', 'a') as f:
    #     for inner_dim in inner_dims:
    #         for lam in lams:
    #             model = MF(num_users, num_items, inner_dim, Y, Y_test, inv_propensities, delta_type, lam)
    #
    #             optimizer = torch.optim.LBFGS(model.parameters())
    #
    #             for epoch in range(EPOCHS):
    #                 optimizer.zero_grad()
    #                 loss = model()
    #                 with torch.no_grad():
    #                     train_err, test_err = model.calc_train_test_err()
    #                     print(f'{epoch+1}. loss: {loss} \t train err: {train_err} \t test err: {test_err} \t lam: {lam} \t inner_dim: {inner_dim} ')
    #                     f.write(f'{epoch+1}. loss: {loss} \t train err: {train_err} \t test err: {test_err} \t lam: {lam} \t inner_dim: {inner_dim}\n')
    #
    #                 if test_err < best_test_err:
    #                     best_test_err = test_err
    #                     best_lam = lam
    #                     best_dim = inner_dim
    #
    #                 def closure():
    #                     optimizer.zero_grad()
    #                     loss = model()
    #                     loss.backward()
    #                     return loss
    #
    #                 optimizer.step(closure)
    #
    # print(f'best test err: {best_test_err} \t lam: {best_lam} \t inner_dim: {best_dim} ')
    #
