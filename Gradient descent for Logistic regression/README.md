# Gradient-descent-for-Logistic-regression

1. gradient_descent.py, data_process_gradient_descent.py

  A gradient descent algorithm for solving the optimization problem.

2. gradient_descent_modified.py, data_process_gradient_descent_modified.py

  Modify your gradient descent code in the following way.

  • Use only the first 0.8n examples to define the objective function; keep the remaining
  n-0.8n examples as a hold-out set.

  • The stopping condition is as follows. After every power-of-two iterations of gradient
  descent, record the hold-out error rate for the linear classifier based on the current
  Bata. If this hold-out error rate is more than 0.99 times that of the best hold-out error
  rate previously computed, and the number of iterations executed is at least 32 (which is
  somewhat of an arbitrary number), then stop.
