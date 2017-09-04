import numpy as np

class regression_function:
    """
    Class definition for getting statistics of the appropriate regression function
    """
    def __init__(self, order=1, string='logistic', features=[], labels=[]):
        """

        Args:
            order (int) : Choose the order of the method to be working with
            string (str) : Choose the type of regression function. Current support is 'logistic'. Will be expanded in future.

        """
        self.method = string
        self.order = order
        self.f_value = 0
        if self.order > 0:
            self.grad_f_value = 0
        if self.order > 1:
            self.hess_f_value = 0
        self.features = features
        self.labels = labels

    @staticmethod
    def prediction(method, x, w):
        """

        Prediction value based on the regression type

        Args:
            method: Regression type. Currently its 'logistic' and 'linear'. To be extended soon
            x: Vector value
            w: Feature value

        """
        if method == 'logistic':
            return 1./(1 + np.exp(-np.dot(x, w)))
        else:
            return np.dot(x,w)

    def function_update(self, x):
        """

        Args:
            x: New vector value for which function value is to be updated

        Returns:
            total_value: Updated function value

        """
        total_value = 0
        for i in range(len(self.labels)):
            total_value += self.labels[i]*np.log(self.prediction(self.method, x, self.features[i])) \
                           + (1-self.labels[i])*np.log(self.prediction(self.method, x, self.features[i]))
        return total_value

    def grad_update(self, x):
        """

        Args:
            x: New vector value for which function value is to be updated

        Returns:
            total_value: Updated gradient vector value

        """
        total_update = 0
        for i in range(len(self.labels)):
            total_update += (self.prediction(self.method, x, self.features[i]) - self.labels[i])*self.features[i]
        return total_update

    def hess_update(self, x):
        """

        Args:
            x: New vector value for which function value is to be updated

        Returns:
            total_value: Updated hessian matrix value

        NOTE: Coming soon

        """
        pass

    def rerun(self, new_x):
        """

        Updates the function and its derevatives values based on the updated x value.

        Args:
            new_x: New vector value for which function value is to be updated

        """
        self.f_value = self.function_update(x = new_x)
        if hasattr(self, 'grad_f_value'):
            self.grad_f_value = self.grad_update(x = new_x)
            if hasattr(self, 'hess_f_value'):
                self.hess_f_value = self.hess_update(x = new_x)