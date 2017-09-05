from __future__ import print_function
import functions
import numpy as np
import argparse

############## INITIALIZATION DATASET######################
# The following dataset is obtained from https://machinelearningmastery.com/implement-logistic-regression-stochastic-gradient-descent-scratch-python/
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]

labels = np.array([i[2] for i in dataset])

features = [i[:2] for i in dataset]
for i in features:
    i.append(1)
features = np.array(features)
#####################################################

def update_x(x_old, x_update, step_size):
    """

    Args:
        x_old: Current vector value to be updated.
        x_update: Update direction (Typically gradient of the Loss functions).
        step_size: Step size for moving in the update direction.

    Returns:
        x_new: New vector value after updating.

    """
    x_new = x_old - step_size*x_update
    return x_new

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Logistic Regression.')
    parser.add_argument('--err_lim', dest='err_lim', type=float,
                        help='error limit for the regression to end')
    parser.add_argument('--step_size', dest='step_size', type=float,
                        help='Step size for gradient update.')
    parser.add_argument('--start_vec', dest='x_start', nargs='+', type=int,
                        help='3-D Initial vector for the regression. For e.g. [1,0,1] will be written as 1 0 1')
    args = parser.parse_args()

    x_old = np.array(args.x_start)
    error_limit = args.err_lim
    step_size = args.step_size

    cls = functions.RegressionClass(features=features, labels=labels)
    epoch = 0
    curr_error = 0

    while(abs(curr_error) > error_limit or epoch == 0):
        cls.rerun(x_old)
        x_new = update_x(x_old=x_old, x_update=cls.grad_f_value, step_size=step_size)

        x_old = x_new
        curr_error = sum([labels[i] - cls.prediction('logistic', x_old, features[i]) for i in range(len(labels))])
        epoch += 1

    print("Error : ", curr_error)
    print("Predicted Lables : ", [cls.prediction('logistic', x_old, i) for i in features])
    print("Actual Labels :", labels)
    print("Coefficients : ", x_old)
    print("Epoch reached is : ", epoch)