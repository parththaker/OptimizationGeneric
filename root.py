from __future__ import print_function
import functions
import numpy as np
import argparse
import data_generator
import methods
import plot_graph
import basic_optimization
import error_functions
import essentials

# These parameteres soon to be shifted to a config file

config_setting = essentials.ConfigSectionMap('config.ini', 'optimize')

datafile = str(config_setting['datafile'])
entries = int(config_setting['entries'])
epoch_limit = int(config_setting['epoch_limit'])

# End of Parameteres

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Optimization Simulation.')
    parser.add_argument('--function_type', dest='ftype', type=str, help='Type of regression to be done. As of now, logistic, linear, diff, ndiff')
    parser.add_argument('--method_type', dest='mtype', type=str, help='Type of regression to be done. As of now, logistic, linear, diff, ndiff')
    parser.add_argument('--err_lim', dest='err_lim', type=float,
                        help='error limit for the regression to end')
    parser.add_argument('--step_size', dest='step_size', type=float,
                        help='Step size for gradient update.')
    parser.add_argument('--plot', action='store_true',help='Enabling argument for plotting')
    parser.add_argument('--stoc', action='store_true',help='Enabling argument for Stochastic oracle')
    args = parser.parse_args()

    function_type = args.ftype
    error_limit = args.err_lim
    step_size = args.step_size
    method_type = args.mtype

    plot = plot_graph.PlottingError()
    optimizer = basic_optimization.BasicOptimization(function_type=function_type, method_type=method_type, is_feature=True, is_stoc=args.stoc)
    error_class = error_functions.ErrorClass(x_opt=optimizer.cls.x_opt, f_opt=optimizer.cls.f_opt)

    epoch = 0
    curr_error = 10**10
    x_error = []
    grad_error = []
    func_error = []

    while(abs(curr_error) > error_limit or epoch == 0):
        x, f, g = optimizer.run(step_size=step_size)

        x_err = error_class.x_error(x)
        x_error.append(x_err)

        f_error, g_error = error_class.function_error(g, f)
        func_error.append(f_error)
        grad_error.append(g_error)

        curr_error=f_error

        if f_error==-1:
            curr_error = g_error

        epoch+=1
        if epoch > epoch_limit:
            print("Epoch count reached limit. \nIncrease the limit and continue.")
            break

    print("Error : ", curr_error)
    print("X : ", x)
    print("Epoch reached is : ", epoch)
    if args.plot:
        plot.plot(color='#0000FF', x_err=x_error, grad_err = grad_error, funct_err=func_error , epoch=range(epoch), alpha = 1.0)
        plot.label(title='Error Plot')
        plot.show()