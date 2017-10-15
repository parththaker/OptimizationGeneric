from __future__ import print_function
import functions
import numpy as np
import argparse
import data_generator
import methods
import matplotlib.pyplot as plt

# These parameteres soon to be shifted to a config file

datafile = 'Dataset/linear_regression.csv'
entries = 10
epoch_limit = 10000
huber_param = 5

# End of Parameteres

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Regression Simulation.')
    parser.add_argument('--type', dest='type', type=str, help='Type of regression to be done. As of now, logistic, linear, diff, ndiff')
    parser.add_argument('--err_lim', dest='err_lim', type=float,
                        help='error limit for the regression to end')
    parser.add_argument('--step_size', dest='step_size', type=float,
                        help='Step size for gradient update.')
    parser.add_argument('--plot', action='store_true',help='Enabling argument for plotting')
    args = parser.parse_args()

    algo_type = args.type
    error_limit = args.err_lim
    step_size = args.step_size

    features, labels = data_generator.get_feature_labels(filename=datafile, entries=entries)

    if algo_type == 'logistic':
        cls = functions.LogisticRegressionClass(features=features, labels=labels)
    elif algo_type == 'linear':
        cls = functions.LinearRegressionClass(features=features, labels=labels)
    elif algo_type == 'diff':
        cls = functions.DiffRegressionClass()
    elif algo_type == 'ndiff':
        cls = functions.NonDiffRegressionClass()
    elif algo_type == 'huber':
        cls = functions.HuberLossClass(alpha=huber_param)
    else:
        print('You entered some wierd "type". Please correct it. Exiting...')
        exit()

    epoch = 0
    curr_error = 0

    dim_vec = features.shape[1]
    met3 = methods.GradDescent(dim=dim_vec)

    x_old = 0

    if args.plot:
        epoch_arr = []
        error_arr = []


    while(abs(curr_error) > error_limit or epoch == 0):
        met3.update(grad_f=cls.grad_update, step_size = step_size, active_nodes=-1)
        x_old = met3.x
        epoch += 1
        curr_error = np.linalg.norm(cls.grad_update(x_old))
        if args.plot and epoch%10 ==0:
            epoch_arr.append(epoch)
            error_arr.append(curr_error)
        if epoch > epoch_limit:
            print("Epoch count reached limit. \nIncrease the limit and continue.")
            break

    print("Error : ", curr_error)
    print("X : ", x_old)
    print("Epoch reached is : ", epoch)

    plt.semilogy(epoch_arr, error_arr)
    plt.title('Gradient descent error plot')
    plt.grid(True)
    plt.xlabel('Epoch count')
    plt.ylabel(r'||$\nabla f$||')
    plt.show()
