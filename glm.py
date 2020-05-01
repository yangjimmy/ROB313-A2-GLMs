from data_utils import load_dataset
import numpy as np
import math
import sys
import time
import matplotlib.pyplot as plt
from scipy.linalg import solve_triangular


class RBF():
    def __init__(self, center, theta):
        """
        @param x: the center point, 
        """
        self.center = center
        self.theta = theta
        if theta==0:
            raise ValueError
    def evaluate(self, x_i):
        diff = np.linalg.norm(np.subtract(self.center,x_i)) # l2 (Frobenius) norm
        return np.exp(np.divide(-1.0*np.power(diff,2),np.array([self.theta])))
    def evaluate_all(self, x):
        vector = np.empty((0,1))
        for item in x:
            val = self.evaluate(item)
            vector = np.vstack((vector,val))
        return vector
    def __str__(self):
        print("the rbf function centered at {} with shape parameter {}".format(self.center,self.theta))



num_functions = 5
# define basis functions #
def _f0(x):
    return 1

def _f1(x):
    return x
    #return 1

def _f2(x):
    return np.power(x,2)
    #return x

# period is around 100
def _f3(x):
    return x*np.sin(111*x)
    #return math.sin(111*x)

def _f4(x):
    return x*np.cos(111*x)
    #return math.cos(111*x)
#########################

### internal methods ###
# create phi vector
def _phi(x):
    return np.array([_f0(x),_f1(x),_f2(x),_f3(x),_f4(x)])
    #return np.array([_f1(x),_f2(x),_f3(x),_f4(x)])

def _SVD(X):
    """
    @param X: matrix (np array)
    @rtype: list of matrices
    """
    U, s, V_t = np.linalg.svd(X, full_matrices=True)
    #S = np.zeros((U.shape[0],V_t.shape[0]))
    #S[:s.shape[0],:s.shape[0]] = np.diag(s)
    S = np.diag(s)
    S = np.pad(S, ((0,U.shape[0]-S.shape[0]),(V_t.shape[0]-S.shape[1],0)),'constant',constant_values=(0))
    return U, S, V_t

def _rmse(y_pred,y):
    """
    returns the rmse error between two sets of y values
    """
    error_tot=0
    difference = np.subtract(y_pred,y)
    for error in range(len(difference)):
        error_tot += difference[error]**2
    error_tot/=len(difference)
    
    return math.sqrt(error_tot)

# create phi matrix using input points x_bar
def _PHI(x):
    """
    @param x: np array of x values of training points
    """
    matrix = np.empty((0,num_functions), float)
    for x_train in np.nditer(x):
        matrix = np.vstack((matrix,_phi(x_train)))
    return matrix

def _plot2(x,y1,y2,legend1,legend2,x_label,y_label,title):
    if type(legend1) is not str or type(legend2) is not str or type(x_label) is not str or type(y_label) is not str or type(title) is not str:
        raise TypeError
    line1,line2 = plt.plot(x,y1,x,y2)
    plt.legend((line1,line2),(legend1,legend2))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.figure()
    plt.show()

def _k(x,z):
    """
    compute the kernel
    """
    return 1 + x*z + np.power(x,2)*np.power(z,2)+x*z*math.cos(111*x)*math.cos(111*z)+x*z*math.sin(111*x)*math.sin(111*z)

def _k_rbf(x,z,theta):
    """
    compute the value of the radial basis function (Gaussian kernel)
    @param x: first vector np.array
    @param z: second vector np.array
    @param theta: shape parameter, float
    """
    temp = np.subtract(x,z)
    diff = np.linalg.norm(temp) # l2 (Frobenius) norm
    #print(type(diff))
    if theta==0:
        print("theta is zero")
    return np.exp(np.divide(-1.0*np.power(diff,2),np.array([theta])))

def _accuracy(y_pred, y):
    diff = np.subtract(y_pred,y)
    inaccurate = np.count_nonzero(diff)
    return (len(y_pred)-inaccurate)/len(y_pred)*100.0

########################

def glm(x_train, y_train, x_valid, y_valid, lambda_start, lambda_end):
    """
    @param x: np array of x values of training points
    @param y: np array of y values of training points
    @param lambda_start: start value of lambda
    @param lambda_end: end value of lambda
    @rval w_optimal: optimal weights
    """
    # initialize
    X_valid = _PHI(x_valid)
    error = float('inf')
    
    # SVD
    temp = _PHI(x_train)
    U,S,V_t = _SVD(temp)
    V = np.transpose(V_t)
    S_t = np.transpose(S)
    U_t = np.transpose(U)

    # find lambda
    l_optimal=lambda_start
    w_optimal=None
    temp0 = np.dot(U_t,y_train)
    temp1 = np.dot(S_t,temp0)
    for l in range(lambda_start, lambda_end+1):
        temp2 = np.linalg.inv(np.dot(S_t,S)+l*np.eye(num_functions))
        w = np.dot(V,np.dot(temp2,temp1))
        y_pred = np.dot(X_valid,w)
        error_new = _rmse(y_pred,y_valid)
        if error_new < error:
            w_optimal = w
            error = error_new
            l_optimal=l
    print("The minimum validation error is {}".format(error))
    print("The regularization parameter corresponding to this error value is {}".format(l_optimal))
    return w_optimal


def glm_test(w, x, y):
    """
    run glm on test and plot results
    @param w: weights (np array)
    @param x: test x values (np array)
    @param y: test (true) y values (np array)
    @rtype: None
    """
    X = _PHI(x)
    y_pred = np.dot(X,w)
    error = _rmse(y_pred,y)
    print("The test error for this GLM model is {}".format(error))
    _plot2(x=x,y1=y,y2=y_pred,legend1="Actual y values",legend2="Predicted y values",x_label="x",y_label="y",title="Mauna Loa Predictions")


def glm_kernelized (x_train, y_train, x_valid, y_valid, x_test, y_test, l=5):
    """
    produce a overview of a kernelized GLM
    includes 1) prediction on a test set
    2) graph of the kernel and its translation
    """
    # construct a larger train dataset (do not need validation, lambda predetermined)
    x = np.vstack((x_train, x_valid))
    y = np.vstack((y_train, y_valid))

    # length of dataset x; N
    N = len(x)

    # construct gram matrix
    K = np.empty((N,N)) # N by N
    for i in range(N):
        for j in range(i+1):
            # using property of kernels, k(a,b) = k(b,a) for all a, b
            k = _k(x[i],x[j])
            K[i,j] = k
            K[j,i] = k
    
    # find cholesky
    L = np.linalg.cholesky(K+l*np.eye(N))

    # construct alpha vector
    z = solve_triangular(L, y, lower=True) # solve the lower triangular matrix
    a = solve_triangular(np.transpose(L),z)

    # find k matrix for running test predictions
    K_test = np.zeros((len(x_test),N))
    for i in range(len(x_test)):
        for j in range(N):
            K_test[i][j] = _k(x_test[i],x[j])
    
    # find the test predictions
    y_pred = np.dot(K_test,a)
    error = _rmse(y_pred,y_test)
    print("the test error is {}".format(error))

    # plot the predicted vs actual
    _plot2(x_test, y_pred, y_test, "predicted values", "actual values", "x values", "y values", "Mauna Loa with kernelized GLM")

    # plot the kernel
    z = np.linspace(-0.1,0.1,100)
    k1 = np.empty((0,1), float)
    k2 = np.empty((0,1), float)
    for i in range(len(z)):
        k1 = np.vstack((k1,_k(0,z[i])))
    plt.ylabel("k")
    plt.xlabel("z")
    plt.title("k vs z")
    plt.figure()
    plt.plot(z, k1)
    for i in range(len(z)):
        k2 = np.vstack((k2,_k(1,z[i]+1)))
    plt.ylabel("k")
    plt.xlabel("z")
    plt.title("k vs z")
    plt.figure()
    plt.plot(z, k2)
    plt.show()

def _glm_rbf_helper (x, y, x_test, y_test, theta, reg=True):
    """
    find the minimum error and corresponding l
    """
    
    # length of dataset x; N
    N = len(x)

    # construct gram matrix
    K = np.empty((N,N)) # N by N
    for i in range(N):
        for j in range(i+1):
            # using property of kernels, k(a,b) = k(b,a) for all a, b
            k = _k_rbf(x[i],x[j], theta)
            K[i,j] = k
            K[j,i] = k
    
    errors = np.array([])
    # loop over all regularization parameter values
    l_all = [0.001,0.01,0.1,1]
    for i in range(len(l_all)):
        print("iteration number: {}".format(i))
        l = l_all[i]
        # find cholesky
        L = np.linalg.cholesky(K+l*np.eye(N))

        # construct alpha vector
        z = solve_triangular(L, y, lower=True) # solve the lower triangular matrix
        a = solve_triangular(np.transpose(L),z)

        # find k matrix for running test predictions
        K_test = np.zeros((len(x_test),N))
        for j in range(len(x_test)):
            for k in range(N):
                
                K_test[j][k] = _k_rbf(x_test[j],x[k],theta)
        
        # find the test predictions
        y_pred = np.dot(K_test,a)

        # find the error
        if reg:
            errors = np.append(errors, [_rmse(y_pred, y_test)],axis=0)
        else:
            # preprocessing
            y_pred_intlabel = np.argmax(y_pred,axis=1)
            y_test_intlabel = np.argmax(y_test,axis=1)
            errors = np.append(errors, [_accuracy(y_pred_intlabel,y_test_intlabel)],axis=0) 

    return errors


def glm_rbf(x_train, y_train, x_valid, y_valid, x_test, y_test, reg=True):
    # construct a larger train dataset (do not need validation, lambda predetermined)
    x = np.vstack((x_train, x_valid))
    y = np.vstack((y_train, y_valid))

    theta_all = [0.05,0.1,0.5,1,2]
    error_tabulated = np.empty((0,4),float)
    for i in range(len(theta_all)):
        theta = theta_all[i]
        error_tabulated = np.vstack((error_tabulated,_glm_rbf_helper(x,y,x_test,y_test,theta,reg)))
    return error_tabulated


def greedy_reg(x_train, y_train, x_valid, y_valid, x_test, y_test, theta):
    """
    finds the set of basis functions
    """
    # construct the set of basis functions with centres
    x = x_train
    y = y_train
    N = len(x)
    # candidate basis functions
    candidates = []
    for val in x:
        candidates.append(RBF(val, theta))
    # selected basis functions
    selected = []
    # keep track of iteration
    k = 0
    # previous MDL -> if MDL starts to increase then we are finished; o/w continue
    MDL_prev = np.Inf
    # current MDL, temporary value
    MDL_curr = np.Inf
    # residual, initially at y
    residual = np.empty((0,1))
    for yval in y:
        residual = np.vstack((residual,yval))
    residual = np.negative(residual)
    # initialize phi matrix
    phi_curr = np.empty((N,0))
    # begin the algorithm
    while True:
        if (k>0 and MDL_curr>MDL_prev) or len(candidates)==0:
            if len(candidates)==0:
                print("empty")
            # break out if error begins to increase or if no more basis functions to add
            break
        k+=1
        phi_vec_selected = None
        fn_selected = None
        j_phi_max = -1
        for basis_fn in candidates:
            # generate phi vector first
            phi_vec = basis_fn.evaluate_all(x)
            # find metric
            j_phi_num = np.power(np.vdot(residual,phi_vec),2)
            j_phi_den = np.vdot(phi_vec, phi_vec)
            j_phi_i = np.divide(j_phi_num,j_phi_den)
            if j_phi_i > j_phi_max:
                j_phi_max=j_phi_i
                phi_vec_selected = phi_vec
                fn_selected = basis_fn
        # update dictionaries
        candidates.remove(fn_selected)
        selected.append(fn_selected)
        # find ith phi matrix
        phi_curr = np.hstack((phi_curr, phi_vec_selected))
        # estimate weights using cholesky
        if phi_curr.shape[0]>phi_curr.shape[1]:
            # overdetermined least squares
            # print("overdetermined")
            w = np.dot(np.linalg.pinv(phi_curr),y)
        else:
            # underdetermined least squares
            # print("underdetermined")
            temp = np.linalg.inv(np.dot(phi_curr, np.transpose(phi_curr)))
            w = np.dot(np.transpose(phi_curr),np.dot(temp, y))
        # find residual
        y_pred = np.dot(phi_curr,w)
        residual = y_pred - y
        # l-2 loss = |phi*w-y|
        l2_loss = 0
        for error in range(residual.shape[0]):
            l2_loss += residual[error]**2
        # calculate MDL
        MDL_prev = MDL_curr
        MDL_curr = N*1.0/2*(np.log(l2_loss))+k*1.0/2*(np.log(N))
    

    selected.pop() # removes last basis function since it's non optimal


    phi_final = np.empty((N,0))
    for basis_fn in selected:
        # generate phi vector first
        phi_vec = basis_fn.evaluate_all(x)
        phi_final = np.hstack((phi_final, phi_vec))
    w = np.dot(np.linalg.pinv(phi_final),y) # find actual w that does not rely on assumption that w(k-1) is unchanging

    error = _test_eval_greedy(x_test, y_test, selected, w)
    sparsity = (k-1)/N*100 # previous iter is best
    print("For a shape parameter of {} the RMSE error is {} and the sparsity is {}".format(theta,error,sparsity))

def _test_eval_greedy(x_test, y_test, basis_fns, w):
    """
    test the results of the greedy algorithm on the testing set
    """
    N = len(x_test)
    phi_curr = np.empty((N,0))
    for basis_fn in basis_fns:
        # generate phi vector first
        phi_vec = basis_fn.evaluate_all(x_test)
        phi_curr = np.hstack((phi_curr, phi_vec))
    y_pred = np.dot(phi_curr,w)
    return _rmse(y_pred, y_test)

def select_question(number):
    xML_train, xML_valid, xML_test, yML_train, yML_valid, yML_test = load_dataset('mauna_loa')
    xRB_train, xRB_valid, xRB_test, yRB_train, yRB_valid, yRB_test = load_dataset('rosenbrock', n_train=200, d=2) #x is pairs of inputs, y is single value
    xIR_train, xIR_valid, xIR_test, yIR_train, yIR_valid, yIR_test = load_dataset('iris') # x is 4 dimensional
    
    if number == 2:
        # Question 2
        print("Question 2 selected")
        lambda_start = 0
        lambda_end = 20
        w = glm(xML_train, yML_train, xML_valid, yML_valid, lambda_start, lambda_end)
        glm_test(w, xML_test, yML_test)
    elif number == 3:
        # Question 3
        print("Question 3 selected")
        print("The visualization of the kernelized GLM is:")
        glm_kernelized(xML_train, yML_train, xML_valid, yML_valid, xML_test, yML_test)
    elif number == 4:
        # Question 4
        print("Question 4 selected")
        print("Tabulating error values... rows are in increasing order of theta, and columns are in increasing order of lambda")
        print("The tabulated error values for Mauna Loa are:")
        print(glm_rbf(xML_train, yML_train, xML_valid, yML_valid, xML_test, yML_test))
        print("The tabulated error values for Rosenbrock are:")
        print(glm_rbf(xRB_train, yRB_train, xRB_valid, yRB_valid, xRB_test, yRB_test))
        print("The tabulated accuracy values for Iris are:")
        print(glm_rbf(xIR_train, yIR_train, xIR_valid, yIR_valid, xIR_test, yIR_test, False))
    elif number == 5:
        # Question 5
        print("Question 5 selected")
        print("The errors in increasing values of theta are:")
        theta_all = [0.01, 0.1, 1]
        for theta in theta_all:
            print("The error for the shape parameter, theta = {} is:".format(theta))
            greedy_reg(xRB_train, yRB_train, xRB_valid, yRB_valid, xRB_test, yRB_test, theta)
    else:
        print("invalid question number argument")

if __name__ == '__main__':
    select_question(3) # replace with 2, 3, 4, 5 to get desired question
    
    
