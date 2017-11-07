from __future__ import print_function
import functions
import numpy as np
import argparse
import data_generator
import methods
import plot_graph

# These parameteres soon to be shifted to a config file

datafile = 'Dataset/linear_regression.csv'
entries = 10
epoch_limit = 10000
huber_param = 5

# End of Parameteres

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Regression Simulation.')
    parser.add_argument('--algo_type', dest='atype', type=str, help='Type of regression to be done. As of now, logistic, linear, diff, ndiff')
    parser.add_argument('--descent_type', dest='dtype', type=str, help='Type of regression to be done. As of now, logistic, linear, diff, ndiff')
    parser.add_argument('--err_lim', dest='err_lim', type=float,
                        help='error limit for the regression to end')
    parser.add_argument('--step_size', dest='step_size', type=float,
                        help='Step size for gradient update.')
    parser.add_argument('--plot', action='store_true',help='Enabling argument for plotting')
    parser.add_argument('--stoc', action='store_true',help='Enabling argument for Stochastic oracle')
    args = parser.parse_args()

    algo_type = args.atype
    error_limit = args.err_lim
    step_size = args.step_size
    descent_type = args.dtype

    features, labels = data_generator.get_feature_labels(filename=datafile, entries=entries)
    cls, dim_vec = functions.get_regression_class(algo_type, features=features, labels=labels, stoc=[0., 5., 0.])
    met3 = methods.get_descent_method(descent_type=descent_type, dim=dim_vec)

    epoch = 0
    curr_error = 0
    x_old = 0

    plot = plot_graph.PlottingError()

    while(abs(curr_error) > error_limit or epoch == 0):
        met3.update(grad_f=cls.grad_update, step_size = step_size)
        x_old = met3.x
        epoch += 1
        plot.update(cls=cls, x=x_old, epoch=epoch)
        curr_error = plot.funct_err[-1]
        if epoch > epoch_limit:
            print("Epoch count reached limit. \nIncrease the limit and continue.")
            break

    print("Error : ", curr_error)
    print("X : ", x_old)
    print("Epoch reached is : ", epoch)
    if args.plot:
        plot.plot(color='#0000FF')
        plot.label(title='Error Plot')
        plot.show()