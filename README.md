# MLproject1

* `report.pdf` describes the process we used from data preparation to validation and summarizes our results

* `implementations.py` contains standard implementations of the required machine learning methods:
   1. `least_squares_GD(y, tx, initial_w, max_iters, gamma)`: Linear regression using gradient descent
   2. `least_squares_SGD(y, tx, initial_w, max_iters, gamma)`: Linear regression using stochastic gradient descent
   3. `least_squares(y, tx)`: Least squares regression using normal equations
   4. `ridge_regression(y, tx, lambda_)`: Ridge regression using normal equations
   5. `logistic_regression(y, tx, initial_w, max_iters, gamma)`: Logistic regression using gradient descent
   6. `reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)`: Regularized logistic regression using gradient descent

* `run.py` is an executable that computes the predictions submitted on aircrowd: should be executed with train dataset at the following path `"data/train.csv"`

* `project1.ipynb` gathers the whole python code we used to explore the dataset, compute and test the implemented machine learning methods

* `proj1_helpers.py`  provided helpers functions