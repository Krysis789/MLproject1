##Implementations of the required function

#Linear regression using gradient descent
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    nIters = 0 #Keep count of iterations
    w = initial_w #Current weights
    n = len(y) #Number of observations
    while (nIters < max_iters):
        e = y - np.dot(tx, w) #Residual vector
        gradient = -np.dot(np.transpose(tx), e) / n #Gradient
        w -= gamma * gradient #A step towards negative gradient
        nIters += 1 #Update number of iterations
    e = y - np.dot(tx, w) #Compute final residuals
    return w, np.dot(np.transpose(e), e) / (2 * n) #Return weights and loss (2n as a scaler)

#Linear regression using stochastic gradient descent
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    nIters = 0 #Keep track of iterations
    w = initial_w #Update w
    n = len(y) #Number of datapoints
    while (nIters < max_iters):
        index = np.random.randint(0, n) #Pick a row uniformly
        row = tx[index, :] #Select the chosen row 
        e = y[index] - np.dot(row, w) #Calculate the estimate for error
        gradient = -np.dot(np.transpose(row), e) #Calculate the estimate for the gradient
        w -= gamma * gradient #Update w
        nIters += 1 #Update number of iterations
    e = y - np.dot(tx, w) #Calculate the residuals for the final loss
    return w, np.dot(np.transpose(e), e) / (2 * n) #Return the weights and the loss (2n as a scaler)

#Least squares regression using normal equations
def least_squares(y, tx):
    xtx = np.dot(np.transpose(tx), tx) #Calculate the Gram matrix
    w = np.dot(np.dot(np.linalg.inv(xtx), np.transpose(tx)), y) #Calculate the weigths
    e = y - np.dot(tx, w) #Calculate the residuals
    loss = np.dot(np.transpose(e), e) / (2 * len(y)) #Calculate the loss (2n as a scaler)
    return w, loss

#Ridge regression using normal equations
def ridge_regression(y, tx, lambda_):
    xtx = np.dot(np.transpose(tx), tx) + lambda_ * np.identity(np.shape(tx)[1]) #Calculate the modified Gram matrix
    w = np.dot(np.dot(np.linalg.inv(xtx), np.transpose(tx)), y) #Calculate the weigths
    e = y - np.dot(tx, w) #Calculate residuals
    loss = np.dot(np.transpose(e), e) / (2 * len(y)) #Calculate the loss (2n as a scaler)
    return w, loss

#A helper for the logistic regression
def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))

#Logistic regression using gradient descent
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    nIters = 0
    w = initial_w
    n = len(y)
    while (nIters < max_iters): 
        gradient = np.dot(np.transpose(tx), sigmoid(np.dot(tx, w)) - y)
        w -= gamma * gradient #Calculate new w
        nIters += 1 #Update number of iterations
    loss = np.sum(np.log(1 + np.exp(np.dot(tx, w))) - y * np.dot(tx, w))
    return w, loss #Return weights and loss

#Regularized logistic regression using gradient descent or SGD
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    #xxxx
    nIters = 0
    w = initial_w
    n = len(y)
    while (nIters < max_iters): 
        ##Calculate the hessian matrix
        S = sigmoid(np.dot(tx, w))
        S = S * (1 - S)
        tmp = np.transpose((np.transpose(tx) * S))
        hessian = np.dot(np.transpose(tx), tmp) + lambda_
        
        #Calculate the gradient
        gradient = np.dot(np.transpose(tx), sigmoid(np.dot(tx, w)) - y) + lambda_ * w
        #Update w
        w -= gamma * np.dot(np.linalg.inv(hessian), gradient)
        #Update number of iterations
        nIters += 1
        
    #Calculate the loss
    loss = np.sum(np.log(1 + np.exp(np.dot(tx, w)))) - np.sum(np.dot(y, np.dot(tx, w))) + lambda_ / 2 * np.linalg.norm(w)
    return w, loss #Return weights and loss
