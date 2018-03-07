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
epoch_limit = 100
huber_param = 5

# End of Parameteres

def aggregate_readings(avg, x, num):
    if avg == []:
        avg = x
    else:
        for i in range(len(x)):
            new_x = float(avg[i]*num + x[i])/(num+1)
            avg[i] = new_x
    return avg


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
    parser.add_argument('--num', dest='num', type=int,help='Number of parallel tracks')
    args = parser.parse_args()

    algo_type = args.atype
    error_limit = args.err_lim
    step_size = args.step_size
    descent_type = args.dtype
    parallel_threads = args.num

    features, labels = data_generator.get_feature_labels(filename=datafile, entries=entries)

    plot = plot_graph.PlottingError(stoc = True)

    x_err = []
    funct_err = []
    grad_err = []

    for i in range(parallel_threads):

        cls, dim_vec = functions.get_regression_class(algo_type, features=features, labels=labels, stoc=[0., 0.5, 0.])
        met3 = methods.get_descent_method(descent_type=descent_type, dim=dim_vec)

        epoch = 0
        curr_error = 0
        x_old = 0

        while(abs(curr_error) > error_limit or epoch == 0):
            met3.update(grad_f=cls.grad_update, step_size = step_size)
            x_old = met3.x
            epoch += 1
            plot.update(cls=cls, x=x_old, epoch=epoch)
            curr_error = plot.funct_err[-1]
            if epoch > epoch_limit:
                print("Epoch count reached limit. \nIncrease the limit and continue.")
                break

        x_err = aggregate_readings(x_err, plot.x_err, i)
        funct_err = aggregate_readings(funct_err, plot.funct_err, i)
        grad_err = aggregate_readings(grad_err, plot.grad_err, i)

        print("Error : ", curr_error)
        print("X : ", x_old)
        print("Epoch reached is : ", epoch)

        if args.plot:
            plot.plot(color='#0000FF')
            plot.refresh()


    plot.plt.figure(1)
    plot.plt.semilogy(range(len(x_err)), x_err, label=r'$\frac{1}{50}\sum_{i=1}^{50}||x_i||$', color='r')

    plot.plt.figure(2)
    plot.plt.semilogy(range(len(grad_err)), grad_err, label=r'$\frac{1}{50}\sum_{i=1}^{50}||\nabla f_i(x)||$', color='r')

    plot.plt.figure(3)
    plot.plt.semilogy(range(len(funct_err)), funct_err, label=r'$\frac{1}{50}\sum_{i=1}^{50}|f_i(x)|$', color='r')

    plot.label(title='Error Plot')

    if args.plot:
        plot.show()

