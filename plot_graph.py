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
epoch_limit = 200
huber_param = 5

# End of Parameteres

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


class PlottingError(object):
    def __init__(self, stoc = False):
        self.plt = plt
        self.alpha = 1.
        if stoc == True:
            self.alpha = 0.2
        self.is_x_err = False
        self.is_func_err = False
        self.is_grad_err = False


    def plot(self, color, x_err, grad_err, funct_err , epoch, alpha = None, label=None):

        if not alpha:
            alpha = self.alpha

        if x_err:
            self.is_x_err=True
            self.plt.figure(1)
            self.plt.semilogy(epoch, x_err, color=color, alpha=alpha, label=label)

        if grad_err:
            self.is_grad_err=True
            self.plt.figure(2)
            self.plt.semilogy(epoch, grad_err, color=color, alpha=alpha, label=label)

        if funct_err:
            self.is_func_err=True
            self.plt.figure(3)
            self.plt.semilogy(epoch, funct_err, color=color, alpha=alpha, label=label)

    def label(self, title):

        if self.is_x_err:
            self.plt.figure(1)
            self.plt.title(title)
            self.plt.grid(True)
            self.plt.legend()
            self.plt.xlabel('Epoch count')
            self.plt.ylabel(r'$||x -x^*||$')

        if self.is_grad_err:
            self.plt.figure(2)
            self.plt.title(title)
            self.plt.grid(True)
            self.plt.legend()
            self.plt.xlabel('Epoch count')
            self.plt.ylabel(r'$||\nabla f(x)||$')

        if self.is_func_err:
            self.plt.figure(3)
            self.plt.title(title)
            self.plt.grid(True)
            self.plt.legend()
            self.plt.xlabel('Epoch count')
            self.plt.ylabel(r'$|f(x) - f^*|$')

    def show(self):
        self.plt.show()

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
        dim_vec = features.shape[1]

    elif algo_type == 'linear':
        cls = functions.LinearRegressionClass(features=features, labels=labels)
        dim_vec = features.shape[1]

    elif algo_type == 'diff':
        cls = functions.DiffRegressionClass()
        dim_vec = 1

    elif algo_type == 'ndiff':
        cls = functions.NonDiffRegressionClass()
        dim_vec = 1

    elif algo_type == 'huber':
        cls = functions.HuberLossClass(alpha=huber_param)
        dim_vec = 1

    elif algo_type == 'camel':
        cls = functions.ThreeHumpCamelFunction()
        dim_vec = 2

    elif algo_type == 'matyas':
        cls = functions.MatyasFunction()
        dim_vec = 2

    elif algo_type == 'bulkin':
        cls = functions.BukinN6Function()
        dim_vec = 2

    elif algo_type == 'booth':
        cls = functions.BoothFunction()
        dim_vec = 2

    elif algo_type == 'trid':
        dim_vec = 10
        cls = functions.TridFunction(dim_vec=dim_vec)

    elif algo_type == 'stoc':
        dim_vec = 10
        cls = functions.StocLossClass(mean= 0., sigma=1., upper=2., constant=0., dim_vec=dim_vec)

    else:
        print('You entered some wierd "type". Please correct it. Exiting...')
        exit()

    epoch = 0
    curr_error = 0

    met3 = methods.StochasticMethod1(dim=dim_vec)

    x_old = 0

    if args.plot:
        epoch_arr = []
        error_arr = []


    while(abs(curr_error) > error_limit or epoch == 0):
        met3.update(grad_f=cls.grad_update, step_size = step_size)
        x_old = met3.x
        epoch += 1
        # curr_error = np.linalg.norm(cls.grad_update(x_old))
        # curr_error = np.abs(cls.function_update(x_old))
        curr_error = np.linalg.norm(x_old)
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

