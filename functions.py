import numpy as np
import math

def grad_norm(x, y):
    return 2*(x-y)

def norm_vec(x,y):
    return np.linalg.norm(x-y)

class LogisticRegressionClass(object):

    def __init__(self, features, labels, order=1):

        self.order = order
        self.f_value = 0
        if self.order > 0:
            self.grad_f_value = 0
        if self.order > 1:
            self.hess_f_value = 0
        self.features = features
        self.labels = labels

    @staticmethod
    def prediction(x, w, label):

        value = 1./(1 + np.exp(-label*np.dot(x, w)))
        return value

    @staticmethod
    def event_probability(x, w):

        value = 1./(1 + np.exp(-np.dot(x, w)))
        return value

    def function_update(self, x):

        total_value = 0
        for label, feature in zip(self.labels, self.features):
            true_pr = self.prediction(x=x, w=feature, label=label)
            total_value += -math.log(true_pr)
        return total_value

    def grad_update(self, x):

        total_update = 0
        for label, feature in zip(self.labels, self.features):
            true_pr = self.prediction(x=x, w=feature, label=label)
            total_update += (1-true_pr)*label*feature
        return -total_update

    def hess_update(self, x):

        pass

    def rerun(self, new_x):

        self.f_value = self.function_update(x = new_x)
        if hasattr(self, 'grad_f_value'):
            self.grad_f_value = self.grad_update(x = new_x)
            if hasattr(self, 'hess_f_value'):
                self.hess_f_value = self.hess_update(x = new_x)


class LinearRegressionClass(object):


    def __init__(self, features, labels, order=1):

        self.order = order
        self.f_value = 0
        if self.order > 0:
            self.grad_f_value = 0
        if self.order > 1:
            self.hess_f_value = 0
        self.features = features
        self.labels = labels

    @staticmethod
    def prediction(x, w):

        return np.dot(x,w)

    def function_update(self, x):

        total_value = 0
        for label, feature in zip(self.labels, self.features):
            true_pr = self.prediction(x=x, w=feature)
            total_value += (true_pr - label)**2
        return total_value

    def grad_update(self, x):

        total_update = 0
        for label, feature in zip(self.labels, self.features):
            true_pr = self.prediction(x=x, w=feature)
            total_update += 2*(true_pr - label)*feature
        return total_update

    def hess_update(self, x):

        pass

    def rerun(self, new_x):

        self.f_value = self.function_update(x=new_x)
        if hasattr(self, 'grad_f_value'):
            self.grad_f_value = self.grad_update(x=new_x)
            if hasattr(self, 'hess_f_value'):
                self.hess_f_value = self.hess_update(x=new_x)

class NonDiffRegressionClass(object):

    def __init__(self, order=1):

        self.order = order
        self.f_value = 0
        if self.order > 0:
            self.grad_f_value = 0

    def function_update(self, x):

        if abs(x) > 1:
            return abs(x)
        else:
            return x**2

    def grad_update(self, x):

        if abs(x) > 1:
            return np.sign(x)
        else:
            return 2*x

    def rerun(self, new_x):

        self.f_value = self.function_update(x=new_x)
        if hasattr(self, 'grad_f_value'):
            self.grad_f_value = self.grad_update(x=new_x)


class DiffRegressionClass(object):

    def __init__(self, order=1):

        self.order = order
        self.f_value = 0
        if self.order > 0:
            self.grad_f_value = 0

    def function_update(self, x):

        if abs(x) > 1:
            return abs(x) - 1./2.
        else:
            return x**2/2

    def grad_update(self, x):

        if abs(x) > 1:
            return np.sign(x)
        else:
            return 2*x

    def rerun(self, new_x):

        self.f_value = self.function_update(x=new_x)
        if hasattr(self, 'grad_f_value'):
            self.grad_f_value = self.grad_update(x=new_x)
