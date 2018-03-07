import math
import numpy as np

class ErrorClass(object):
    def __init__(self, x_opt=None, f_opt=None):
        self.is_x_opt = False
        self.is_f_opt = False
        self.f_opt = None
        self.x_opt = None

        if f_opt!=None:
            self.f_opt = f_opt
            self.is_f_opt = True
            self.x_opt = x_opt
            self.is_x_opt = True

    def function_error(self, grad_f, f):
        error_2 = np.linalg.norm(grad_f)
        error_1 = -1
        if self.is_f_opt:
            error_1 = np.abs(f - self.f_opt)
        return [error_1, error_2]

    def x_error(self, x):
        x_error = -1
        if self.is_x_opt:
            x_error = np.linalg.norm(x - self.x_opt)
        return x_error