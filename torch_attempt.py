import torch.nn as nn
import torch
from reg_numpy_attempt import read_yahoo

class MF(nn.Module):
    def __init__(self, num_users, num_items, inner_dim, Y, Y_test, inv_propensities, delta_type, lam):
        super(MF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
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

    def forward(self):
        if self.delta_type == "MSE":
            scores = self.V @ self.W.T+self.a.unsqueeze(-1)+self.b+self.c
            diff = (self.Y - scores)**2
        else:
            scores = self.V @ self.W.T + self.a.unsqueeze(-1) + self.b + self.c
            diff = torch.abs(self.Y - scores)
        regularization = torch.norm(self.V)**2 + torch.norm(self.W)**2 + torch.norm(self.a)**2
        regularization += torch.norm(self.b)**2 + torch.norm(self.c)**2

        diff = self.inv_propensities * diff

        objective = diff.sum() + self.lam * regularization

        return objective

    def calc_train__test_err(self):
        scores = self.V @ self.W.T+self.a.unsqueeze(-1)+self.b+self.c

        if self.delta_type == "MSE":
            diff_train = (self.Y[self.Y != 0] - scores[self.Y != 0])**2
            diff_test = (self.Y_test[self.Y_test != 0] - scores[self.Y_test != 0]) ** 2
        else:
            diff_train = torch.abs(self.Y[self.Y != 0] - scores[self.Y != 0])
            diff_test = torch.abs(self.Y_test[self.Y_test != 0] - scores[self.Y_test != 0])

        return diff_train.mean(), diff_test.mean()


if __name__ == '__main__':
    Y, Y_test, inv_propensities = read_yahoo(path="data/yahoo_data")
    EPOCHS = 10
    num_users, num_items = Y.shape
    inner_dims = [5, 10, 20, 40]
    lams = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    delta_type = 'MSE'
    Y = torch.from_numpy(Y)
    Y_test = torch.from_numpy(Y_test)
    inv_propensities = torch.from_numpy(inv_propensities)
    lam = 1.
    best_test_err = float('inf')
    best_lam = lams[0]
    best_dim = inner_dims[0]
    with open('torch_find_params.txt', 'a') as f:
        for inner_dim in inner_dims:
            for lam in lams:
                model = MF(num_users, num_items, inner_dim, Y, Y_test, inv_propensities, delta_type, lam)

                optimizer = torch.optim.LBFGS(model.parameters())

                for epoch in range(EPOCHS):
                    optimizer.zero_grad()
                    loss = model()
                    with torch.no_grad():
                        train_err, test_err = model.calc_train__test_err()
                        print(f'{epoch+1}. loss: {loss} \t train err: {train_err} \t test err: {test_err} \t lam: {lam} \t inner_dim: {inner_dim} ')
                        f.write(f'{epoch+1}. loss: {loss} \t train err: {train_err} \t test err: {test_err} \t lam: {lam} \t inner_dim: {inner_dim}\n')

                    if test_err < best_test_err:
                        best_test_err = test_err
                        best_lam = lam
                        best_dim = inner_dim

                    def closure():
                        optimizer.zero_grad()
                        loss = model()
                        loss.backward()
                        return loss

                    optimizer.step(closure)

