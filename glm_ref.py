
import numpy as np
import math
from matplotlib import pyplot as plt
from data_utils import load_dataset

possible_datasets = ["mauna_loa", "rosenbrock", "iris"]
osc_freq = 111.2067         # rad/sec, calculated frequency of the mauna_loa oscillations

def load_data(dataset):
    '''
    load the data from the mauna_loa set to create a GLM from
    '''
    if dataset == 'rosenbrock':
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('rosenbrock', n_train=1000, d=2)
    else:
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(str(dataset))
    return x_train, x_valid, x_test, y_train, y_valid, y_test

def l_2_norm(x1, x2):
    '''
    compute the l-2 norm of a vector x2 with vector x1
    note: if the vectors are different lenght (which they shouldn't be),
        we take the smallest length of vector
    '''
    return np.linalg.norm([x1-x2], ord=2)

def rmse(y_estimates, y_valid):
    '''
    calculate the root mean squared error between the estimated y values and 
        the actual y values
    param y_estimates: list of ints
    param y_valid: list of ints, actual y values
    return: float, rmse value between two lists
    '''
    return np.sqrt(np.average(np.abs(y_estimates-y_valid)**2))

##################### helper basis functions ####################
def x(x):
    return math.sqrt(2) * x
def xsq(x):
    return np.power(x, 2)
def xsinx(x):
    return x * np.sin(osc_freq*x)
def xcosx(x):
    return x * np.cos(osc_freq*x)
#################################################################

def glm_svd_validation(x_train, x_valid, y_train, y_valid, lambda_test = None):
    '''
    run the singular value decomposition for the mauna loa 
    param data: all the training and validation data for validation testing
    param lambda_test: list of ints, lambda values to test GLM on
    return lambda (int), w ()
    '''
    basis_functions = [x, xsq, xsinx, xcosx]         # does not include first entry (1)
    if not lambda_test:
        lambda_test = range(0, 30)

    # make the phi matrix M * N
    phi_matrix = np.ones((len(x_train), 1))
    phi_matrix_validation = np.ones((len(x_valid), 1))

    for f in basis_functions:
        # stack column by column the basis functions evaluated for x_train and x_valid
        phi_matrix = np.hstack([phi_matrix, f(x_train)])
        phi_matrix_validation = np.hstack([phi_matrix_validation, f(x_valid)])

    U, s, V = np.linalg.svd(phi_matrix, full_matrices=True)
    # create the sigma matrix for full SVD by filling it in with zeros
    Sigma = np.vstack([np.diag(s), np.zeros((len(x_train) - len(s), len(s)))])

    # compute the RMSE values, by first calculating the weights, then plugging in the
    # validation data to compare to the actual y values
    min_rmse = np.inf
    min_lambda = -1
    for lamb in lambda_test:
        # sigma T * sigma
        temp = np.dot(Sigma.T, Sigma)
        # (sigma T sigma + lambda 1) ^-1
        temp2 = np.linalg.pinv(temp + lamb*np.eye(len(temp)))
        weights = np.dot(V.T, np.dot(temp2, np.dot(Sigma.T, np.dot(U.T, y_train))))
        y_pred = np.dot(phi_matrix_validation, weights)
        cur_rmse = rmse(y_pred, y_valid)

        # check to see if we need to update minimum lambda
        if cur_rmse < min_rmse:
            min_rmse = cur_rmse
            min_lambda = lamb

    return min_lambda

def glm_svd_test(x_train, x_valid, x_test, y_train, y_valid, y_test, lamb):
    '''
    run the singular value decomposition for the mauna_loa test set,
        using the optimal parameters determined by the glm_svd_validation
        function
    param data: all data for mauna loa set
    param lambda: int, optimal regularization parameter found in validation
    return float, RMSE of the test data using the optimal lambda
    '''
    basis_functions = [x, xsq, xsinx, xcosx]         # does not include first entry (1)
    x_train = np.vstack([x_train, x_valid])
    y_train = np.vstack([y_train, y_valid])

    # make the phi and the phi test matrix, because we need to recompute the weights
    phi_matrix = np.ones((len(x_train), 1))
    phi_matrix_test = np.ones((len(x_test), 1))

    for f in basis_functions:
        # stack column by column the basis functions evaluated for x_train and x_valid
        phi_matrix = np.hstack([phi_matrix, f(x_train)])
        phi_matrix_test = np.hstack([phi_matrix_test, f(x_test)])

    U, s, V = np.linalg.svd(phi_matrix, full_matrices=True)
    # create the sigma matrix for full SVD by filling it in with zeros
    Sigma = np.vstack([np.diag(s), np.zeros((len(x_train) - len(s), len(s)))])

    ####### calculate the RMSE and the predictions #######
    # sigma T . sigma
    temp = np.dot(Sigma.T, Sigma)
    # (sigma T sigma + lambda 1) ^-1
    temp2 = np.linalg.pinv(temp + lamb*np.eye(len(temp)))
    weights = np.dot(V.T, np.dot(temp2, np.dot(Sigma.T, np.dot(U.T, y_train))))
    y_pred = np.dot(phi_matrix_test, weights)
    cur_rmse = rmse(y_pred, y_test)

    # now plot the predictions y_pred and the actual y y_test
    plt.figure(1)
    plt.plot(x_test, y_test, '-b', label='Y Actual')
    plt.plot(x_test, y_pred, '-r', label='Y Predictions')
    plt.xlabel('X Test')
    plt.ylabel('Y')
    plt.title('Mauna Loa GLM predictions for lambda=' + str(lamb))
    plt.legend(loc="lower right")
    plt.savefig('mauna_loa_glm_pred.png')

    return cur_rmse

################ kernel function ####################
def kernel(x, z):
    '''
    see report for derivation of the kernel function
    param x, z: data points
    function is (1 + xz)^2 + xz * cos(freq * (x-y))
    '''
    return (1 + x*z) ** 2 + x*z*math.cos(osc_freq*(x-z))

def glm_kernelized(x_train, x_valid, x_test, y_train, y_valid, y_test, lamb):
    '''
    compute the kernelized GLM model for the basis functions used in q1
        also use the optimal lambda found from part 1
    param data: all mauna loa data
    param lamb: int, optimal lambda value found from part 1
    return float, rmse value using the dual persective GLM
    '''
    basis_functions = [x, xsq, xsinx, xcosx]         # does not include first entry (1)
    x_train = np.vstack([x_train, x_valid])
    y_train = np.vstack([y_train, y_valid])

    # when we compute the kernels, save them in the saved_k dictionary since the 
    # gram matrix is symmetric
    gram = np.empty((len(x_train), len(x_train)))
    saved_k = {}
    for i in range(len(x_train)):
        for j in range(len(x_train)):
            # compute the kernel for each permutation of training points
            if (i, j) in saved_k:
                gram[i, j] = saved_k[(i, j)]
            else:
                val = kernel(x_train[i], x_train[j])
                gram[i, j] = val
                # save val for symmetric half
                saved_k[(j, i)] = val

    # now compute the R using the cholesky decomposition of the gram matrix + lambda (eye)
    R = np.linalg.cholesky((gram + lamb*np.eye(len(gram))))
    # so R * R^T gives you the cholesky decomposition
    R_inverse = np.linalg.inv(R)            # R is lower triangular, so this is easy to compute
    alpha = np.dot(np.dot(R_inverse.T, R_inverse), y_train)

    # save all the k vectors as rows in the k_matrix, so we can just dot product with
    # the alpha value
    k_matrix = np.empty((len(x_test), len(x_train)))
    for i in range(len(x_test)):
        new_row = np.ndarray((len(x_train)))
        for j in range(len(x_train)):
            new_row[j] = kernel(x_test[i], x_train[j])
        k_matrix[i, :] = new_row

    y_pred = np.dot(k_matrix, alpha)
    cur_rmse = rmse(y_pred, y_test)

    plt.figure(2)
    plt.plot(x_test, y_test, '-b', label='Y Actual')
    plt.plot(x_test, y_pred, '-r', label='Y Predictions')
    plt.xlabel('X Test')
    plt.ylabel('Y')
    plt.title('Mauna Loa GLM dual perspective for lambda=' + str(lamb))
    plt.legend(loc="lower right")
    plt.savefig('mauna_loa_glm_dual_pred.png')

    return cur_rmse

def visualize_kernel():
    '''
    function to visualize the kernel that was designed for no dataset in
        particular, but just calls the function kernel for k(i, z+i) and
        graphs the output across z = [-0.1, 0.1]
    '''
    colors = ['g', 'm']
    for i in range(2):

        z = np.linspace(-0.1 + i, 0.1 + i, 100)

        kernel_res = np.ndarray((len(z), 1))
        for j in range(len(z)):
            kernel_res[j] = kernel(i, z[j])

        plt.figure(3 + i)
        plt.plot(z, kernel_res, '-' + str(colors[i]), label='k(' + str(i) + ', z+' + str(i) + ')')
        plt.title('Plot of the kernel function over z = [-0.1, 0.1]')
        plt.xlabel('Z')
        plt.ylabel('Kernel Output')
        plt.legend(loc='lower right')
        plt.savefig('kernel_' + str(i) + '.png')

################ radial basis function ####################
def gaussian_rbf(x, z, theta):
    '''
    the function takes in vectors x and z and returns a float, the 
        gaussian RBF of the two vectors. 
    param x, z, theta: params in the gaussian rbf
        order of x and z does not matter
    '''
    return math.exp(-(l_2_norm(x, z) ** 2) / theta)

def glm_rbf_validation(x_train, x_valid, y_train, y_valid, regression=True):
    '''
    constructs the optimal model for a dataset using a Gaussian Radial Basis Function
    
    param data: from load_data()
    param regression: bool, set to True if working on a regression set, 
        determines the calculation of RMSE or percentage accuracy
    return: oprtimal theta, optimal lambda, validation RMSE or accuracy
    '''
    thetas = [0.05, 0.1, 0.5, 1, 2]           # lengthscales
    lambdas = [0.001, 0.01, 0.1, 1]         # regularization parameter

    results = {}
    for theta in thetas:
        # when we compute the kernels, save them in the saved_k dictionary since the 
        # gram matrix is symmetric
        gram = np.empty((len(x_train), len(x_train)))
        saved_k = {}
        for i in range(len(x_train)):
            for j in range(len(x_train)):
                # compute the kernel for each permutation of training points
                if (i, j) in saved_k:
                    gram[i, j] = saved_k[(i, j)]
                else:
                    val = gaussian_rbf(x_train[i], x_train[j], theta)
                    gram[i, j] = val
                    # save val for symmetric half
                    saved_k[(j, i)] = val

        # make k matrix (gram) with validation data, so we can dot with alpha
        # to get all predictions at once
        k_matrix = np.empty((len(x_valid), len(x_train)))
        for i in range(len(x_valid)):
            new_row = np.ndarray((len(x_train)))
            for j in range(len(x_train)):
                new_row[j] = gaussian_rbf(x_valid[i], x_train[j], theta)
            k_matrix[i, :] = new_row

        # for all lambdas, compute the cholesky decomp. then compute predictions and RMSE
        for lamb in lambdas:
            # now compute the R using the cholesky decomposition of the gram matrix + lambda (eye)
            R = np.linalg.cholesky((gram + lamb*np.eye(len(gram))))
            # so R * R^T gives you the cholesky decomposition
            R_inverse = np.linalg.inv(R)            # R is lower triangular, so this is easy to compute
            alpha = np.dot(np.dot(R_inverse.T, R_inverse), y_train)

            if regression:
                y_valid_pred = np.dot(k_matrix, alpha)
                cur_rmse = rmse(y_valid_pred, y_valid)
                # save results in dict
                results[(theta, lamb)] = cur_rmse
            else:
                y_valid_pred = np.argmax(np.dot(k_matrix, alpha), axis=1)
                y_valid_new = np.argmax(1 * y_valid, axis=1)
                # compute the percentage of correct predictions and store in the results
                results[(theta, lamb)] = (y_valid_pred == y_valid_new).sum() / len(y_valid_new)

    if regression:
        # looking for lowest RMSE
        sorted_res = sorted(results.items(), key=lambda x: x[1])
        opt_theta = sorted_res[0][0][0]
        opt_lambda = sorted_res[0][0][1]
        valid_rmse = sorted_res[0][1]
        return opt_theta, opt_lambda, valid_rmse
    else:
        # looking for highest percentage accuracy
        sorted_res = sorted(results.items(), key=lambda x: x[1], reverse=True)
        opt_theta = sorted_res[0][0][0]
        opt_lambda = sorted_res[0][0][1]
        valid_accuracy = sorted_res[0][1]
        return opt_theta, opt_lambda, valid_accuracy

def glm_rbf_test(x_train, x_valid, x_test, y_train, y_valid, y_test, theta, lamb, regression=True):
    '''
    takes the optimal theta and lambda values from the model created from the validation
        set and executes the Cholesky decomposition of K and predictions based on k alpha
    param data: from load_data()
    param theta, lambda: optimal values calculated in glm_rbf_validation()
    param regression: bool, set to True if working on a regression set, 
        determines the calculation of RMSE or percentage accuracy
    return: test rmse/accuracy
    '''
    x_train = np.vstack([x_train, x_valid])
    y_train = np.vstack([y_train, y_valid])

    # when we compute the kernels, save them in the saved_k dictionary since the 
    # gram matrix is symmetric
    gram = np.empty((len(x_train), len(x_train)))
    saved_k = {}
    for i in range(len(x_train)):
        for j in range(len(x_train)):
            # compute the kernel for each permutation of training points
            if (i, j) in saved_k:
                gram[i, j] = saved_k[(i, j)]
            else:
                val = gaussian_rbf(x_train[i], x_train[j], theta)
                gram[i, j] = val
                # save val for symmetric half
                saved_k[(j, i)] = val

    # make k matrix (gram) with validation data, so we can dot with alpha
    # to get all predictions at once
    k_matrix = np.empty((len(x_test), len(x_train)))
    for i in range(len(x_test)):
        new_row = np.ndarray((len(x_train)))
        for j in range(len(x_train)):
            new_row[j] = gaussian_rbf(x_test[i], x_train[j], theta)
        k_matrix[i, :] = new_row

    # now compute the R using the cholesky decomposition of the gram matrix + lambda (eye)
    R = np.linalg.cholesky((gram + lamb*np.eye(len(gram))))
    # so R * R^T gives you the cholesky decomposition
    R_inverse = np.linalg.inv(R)            # R is lower triangular, so this is easy to compute
    alpha = np.dot(np.dot(R_inverse.T, R_inverse), y_train)

    if regression:
        y_pred = np.dot(k_matrix, alpha)
        cur_rmse = rmse(y_pred, y_test)
        # save results in dict
        result = cur_rmse
    else:
        y_pred = np.argmax(np.dot(k_matrix, alpha), axis=1)
        y_test_new = np.argmax(1 * y_test, axis=1)
        # compute the percentage of correct predictions and store in the results
        result = (y_pred == y_test_new).sum() / len(y_test_new)

    return result

if __name__ == "__main__":
    # main testing block
    q1 = False
    q2 = True
    q3 = False
    # q4 and q5 were done by hand

    ################### question 1 ##########################
    if q1:
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_data('mauna_loa')
        optimal_lambda = glm_svd_validation(x_train, x_valid, y_train, y_valid)
        q1_rmse = glm_svd_test(x_train, x_valid, x_test, y_train, y_valid, y_test, optimal_lambda)
        print('Q1: RMSE val for optimal lambda=' + str(optimal_lambda) + ' is ' + str(q1_rmse))

    ################### question 2 ##########################
    if q2:
        if not q1:      # need to load data and compute the optimal lambda to pass through
                        # to the kernelized GLM function
            x_train, x_valid, x_test, y_train, y_valid, y_test = load_data('mauna_loa')
            optimal_lambda = glm_svd_validation(x_train, x_valid, y_train, y_valid)
        q2_rmse = glm_kernelized(x_train, x_valid, x_test, y_train, y_valid, y_test, optimal_lambda)
        print('Q2: RMSE val for optimal lambda=' + str(optimal_lambda) + ' is ' + str(q2_rmse))
        visualize_kernel()

    ################### question 3 ##########################
    if q3:
        # run GLM with gaussian RBF for the two regression datasets
        for dataset in possible_datasets[0:2]:
            x_train, x_valid, x_test, y_train, y_valid, y_test = load_data(dataset)
            optimal_theta, optimal_lambda, valid_rmse = glm_rbf_validation(x_train, x_valid, y_train, y_valid)
            test_rmse = glm_rbf_test(x_train, x_valid, x_test, y_train, y_valid, y_test, optimal_theta, optimal_lambda)
            print('Q3: ' + str(dataset))
            print('Optimal theta=' + str(optimal_theta) + ' optimal lambda=' + str(optimal_lambda))
            print('Validation RMSE: ' + str(valid_rmse) + ' test RMSE: ' + str(test_rmse))
        
        # perform the same steps for the one classification set
        dataset = "iris"
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_data(dataset)
        optimal_theta, optimal_lambda, valid_accuracy = glm_rbf_validation(x_train, x_valid, y_train, y_valid, False)
        test_accuracy = glm_rbf_test(x_train, x_valid, x_test, y_train, y_valid, y_test, optimal_theta, optimal_lambda, False)
        print('Q3: ' + str(dataset))
        print('Optimal theta=' + str(optimal_theta) + ' optimal lambda=' + str(optimal_lambda))
        print('Validation accuracy: ' + str(valid_accuracy) + ' test accuracy: ' + str(test_accuracy))


