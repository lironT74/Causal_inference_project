from auxiliary import torch, nn

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



if __name__ == '__main__':
    pass