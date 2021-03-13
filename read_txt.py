
if __name__ == '__main__':
    with open('torch_find_params_mse.txt', 'r')as file:
        for index, line in enumerate(file):
            splitted_line = line.split()[1:]
            test_err = splitted_line[7]
            lam = splitted_line[9]
            inner_dim = splitted_line[11]
            if index == 0 or test_err < best_test_err:
                best_test_err = test_err
                best_lam  = lam
                best_dim = inner_dim

        print(f'best test err: {best_test_err} \t lam: {best_lam} \t inner_dim: {best_dim} ')