
if __name__ == '__main__':
    with open('test_error_MSE.txt', 'r')as file:
        err_lst = []
        best_err = float('inf')
        for index, line in enumerate(file):
            splitted_line = line.split()
            test_err = float(splitted_line[14])
            if test_err < best_err:
                best_err = test_err
            if (index + 1) % 10 == 0:
                err_lst.append(best_err)
                best_err = float('inf')
        print('MSE test error: ', sum(err_lst)/len(err_lst))

    with open('test_error_MAE.txt', 'r')as file:
        err_lst = []
        best_err = float('inf')
        for index, line in enumerate(file):
            splitted_line = line.split()
            test_err = float(splitted_line[14])
            if test_err < best_err:
                best_err = test_err
            if (index + 1) % 10 == 0:
                err_lst.append(best_err)
                best_err = float('inf')
            if (index + 1) % 50 == 0:
                break
        print('MAE test error: ', sum(err_lst)/len(err_lst))