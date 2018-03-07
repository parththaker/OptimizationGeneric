from __future__ import print_function
import functions
import numpy as np
import data_generator
import methods
import graph_generator
import essentials

# These parameteres soon to be shifted to a config file

config_setting = essentials.ConfigSectionMap('/Users/parththaker/Dropbox/Coding/OptimizationGeneric/config.ini', 'optimize')

datafile = str(config_setting['datafile'])
entries = int(config_setting['entries'])
epoch_limit = int(config_setting['epoch_limit'])

# End of Parameteres

def grad_norm(x, y):
    return 2*(x-y)

def norm_vec(x,y):
    return np.linalg.norm(x-y)

class BasicOptimization(object):
    def __init__(self, function_type, method_type, is_feature=True, is_stoc=False, stoc_params=[0. ,0. ,0.]):
        self.features = None
        self.labels = None
        self.stoc = [0.,0.,0.]

        if is_feature:
            self.features, self.labels = data_generator.get_feature_labels(filename=datafile, entries=entries)

        if is_stoc:
            self.stoc = stoc_params

        self.cls, self.dim_vec = functions.get_regression_class(function_type, features=self.features,
                                                                labels=self.labels, stoc = self.stoc)
        self.method = methods.get_descent_method(descent_type=method_type, dim=self.dim_vec, x=np.array([-10. for i in range(self.dim_vec)]))
        self.epoch = 0
        self.x = -1 # -1 is an indication that x is not set.
        self.f = -1 # -1 is an indication that x is not set.


    def run(self, step_size):
        self.method.update(grad_f=self.cls.grad_update, step_size = step_size)
        self.x = self.method.x
        self.epoch += 1
        self.f = self.cls.function_update(self.x)
        self.g = self.cls.grad_update(self.x)

        return self.x, self.f, self.g

class DistributedOptimization(object):
    def __init__(self, function_type, method_type,  graph_type, nodes,is_feature=True):
        self.features = None
        self.labels = None
        self.nodes = nodes
        self.nodelist = range(nodes)

        if is_feature:
            self.features, self.labels = data_generator.get_feature_labels(filename=datafile, entries=entries)

        self.cls, self.dim_vec = functions.get_regression_class(function_type, features=self.features,
                                                                labels=self.labels)
        self.method = methods.get_descent_method(descent_type=method_type, dim=self.dim_vec, nodes=self.nodes)

        self.graph = graph_generator.GenerateGraph.generate_graph(nodes=self.nodes, type=graph_type)

        self.epoch = 0
        self.x = np.array([np.zeros(self.dim_vec)] * nodes) # -1 is an indication that x is not set.
        self.f = np.array([0] * nodes) # -1 is an indication that x is not set.q

    def update_active_nodes(self, num):
        if num in self.nodelist:
            self.nodelist.remove(num)
        else:
            print("The %i does not exist in nodelist"%(num))
        return 0

    def run(self, step_size):
        self.method.update(grad_f=self.cls.grad_update, grad_gy=grad_norm, grad_gz=grad_norm, edge_matrix=self.graph, step_size=step_size, active_nodes=self.nodelist)
        self.x = self.method.x
        self.f = [self.cls.function_update(self.x[node]) for node in range(self.nodes)]
        self.g = [self.cls.grad_update(self.x[node]) for node in range(self.nodes)]

        return self.x, self.f, self.g