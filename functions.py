import numpy as np
import math

def grad_norm(x, y):
    return 2*(x-y)

def norm_vec(x,y):
    return np.linalg.norm(x-y)

def get_regression_class(algo_type, features=None, labels=None, string='constant', stoc=[0., 0., 0.]):

    if algo_type == 'logistic':
        cls = LogisticRegressionClass(features=features, labels=labels, stoc=stoc)
        dim_vec = features.shape[1]

    elif algo_type == 'linear':
        cls = LinearRegressionClass(features=features, labels=labels, stoc=stoc)
        dim_vec = features.shape[1]

    elif algo_type == 'diff':
        cls = DiffRegressionClass()
        dim_vec = 1

    elif algo_type == 'simple':
        cls = SimpleClass()
        dim_vec = 1

    elif algo_type == 'saddle':
        cls = SimpleSaddleClass()
        dim_vec = 2

    elif algo_type == 'monkey':
        cls = MonkeySaddleClass()
        dim_vec = 2

    elif algo_type == 'ndiff':
        cls = NonDiffRegressionClass()
        dim_vec = 1

    elif algo_type == 'camel':
        cls = ThreeHumpCamelFunction()
        dim_vec = 2

    elif algo_type == 'matyas':
        cls = MatyasFunction()
        dim_vec = 2

    elif algo_type == 'bulkin':
        cls = BukinN6Function()
        dim_vec = 2

    elif algo_type == 'booth':
        cls = BoothFunction()
        dim_vec = 2

    elif algo_type == 'trid':
        dim_vec = 10
        cls = TridFunction(dim_vec=dim_vec)

    elif algo_type == 'stoc':
        dim_vec = 10
        cls = StocLossClass(mean=stoc[0], sigma=stoc[1], upper=stoc[2], constant=0., rule=string, dim_vec=dim_vec)

    else:
        print('You entered some wierd "type". Please correct it. Exiting...')
        exit()
    return cls, dim_vec

class LogisticRegressionClass(object):

    def __init__(self, features, labels, stoc=[0., 0., 0.]):
        self.f_opt = None
        self.x_opt = None
        self.mean= stoc[0]
        self.sigma= stoc[1]
        self.features = features
        self.labels = labels

    def prediction(self, x, w, label):

        value = 1./(1 + np.exp(-label*np.dot(x, w)+ (self.sigma*np.random.randn() + self.mean)))
        return value

    @staticmethod
    def event_probability(x, w):

        value = 1./(1 + np.exp(-np.dot(x, w)))
        return value

    def function_update(self, x):

        total_value = 0
        for label, feature in zip(self.labels, self.features):
            true_pr = self.prediction(x=x, w=feature, label=label)
            total_value += -math.log(true_pr) + 0.5*(np.linalg.norm(x)**2)
        return total_value

    def grad_update(self, x):

        total_update = 0
        for label, feature in zip(self.labels, self.features):
            true_pr = self.prediction(x=x, w=feature, label=label)
            total_update += (1-true_pr)*label*feature - x
        return -total_update

    def hess_update(self, x):

        pass

class LogisticRegressionClass2(object):

    def __init__(self, features, labels, stoc=[0., 0., 0.]):
        self.f_opt = None
        self.x_opt = None
        self.mean= stoc[0]
        self.sigma= stoc[1]
        self.features = features
        self.labels = labels

    def prediction(self, x, w, label):

        value = 1./(1 + np.exp(-label*np.dot(x, w)+ (self.sigma*np.random.randn() + self.mean)))
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


class LinearRegressionClass(object):


    def __init__(self, features, labels, stoc=[0., 0., 0.]):
        self.f_opt = None
        self.x_opt = None
        self.mean = stoc[0]
        self.sigma = stoc[1]
        self.features = features
        self.labels = labels

    def prediction(self, x, w):

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

class NonDiffRegressionClass(object):

    def __init__(self):
        self.f_opt = 0
        self.x_opt = 0

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


class MonkeySaddleClass(object):

    def __init__(self):

        self.f_opt = -2
        self.x_opt = np.array([2, 2])

    def function_update(self, x):

        f = 0

        if x[1] < -1.:
            f += (x[1] +1)**2
        elif x[1] < 1:
            f += 0
        elif x[1] < 3:
            f += (x[1]-2)**2 - 1

        if x[0] < -1.:
            f += (x[0] +1)**2
        elif x[0] < 1:
            f += 0
        elif x[0] < 3:
            f += (x[0]-2)**2 - 1

        return f

    def grad_update(self, x):

        grad = np.array([0., 0.])

        if x[1] < -1.:
            grad[1] = 2*(x[1] +1)
        elif x[1] < 1:
            grad[1] = 0
        else:
            grad[1] = 2*(x[1]-2)

        if x[0] < -1.:
            grad[0] = 2*(x[0] +1)
        elif x[1] < 1:
            grad[0] = 0
        else:
            grad[0] = 2*(x[0]-2)

        return grad


class SimpleSaddleClass(object):

    def __init__(self):

        self.f_opt = -2
        self.x_opt = np.array([2, 2])

    def function_update(self, x):

        f = 0

        if x[1] < 1.:
            f += (x[1] -1)**2
        else:
            f += (x[1]-2)**2 - 1

        if x[0] < 1.:
            f += (x[0] -1)**2
        else:
            f += (x[0]-2)**2 - 1

        return f

    def grad_update(self, x):

        grad = np.array([0., 0.])

        if x[1] < 1.:
            grad[1] = 2*(x[1] -1)
        else:
            grad[1] = 2*(x[1]-2)

        if x[0] < 1.:
            grad[0] = 2*(x[0] -1)
        else:
            grad[0] = 2*(x[0]-2)

        return grad


class SimpleClass(object):

    def __init__(self):
        self.f_opt = 0
        self.x_opt = 10

    def function_update(self, x):

        return (x-10)**2

    def grad_update(self, x):

        return 2*(x-10)


class DiffRegressionClass(object):

    def __init__(self):
        self.f_opt = 0
        self.x_opt = 0

    def function_update(self, x):

        if abs(x) > 1:
            return abs(x) - 1./2.
        else:
            return x**2/2

    def grad_update(self, x):

        if abs(x) > 1:
            return np.sign(x)
        else:
            return x

class StocLossClass(object):

    def __init__(self, mean, sigma, upper, constant, dim_vec, rule='constant'):

        self.sigma = sigma
        self.upper = upper
        self.mean = mean
        self.constant = constant
        self.conditon_function = self.get_rule_function(rule)
        self.f_opt = 0
        self.x_opt = np.array([0]*dim_vec)

    def get_rule_function(self, string):
        if string == 'constant':
            func = lambda i : 1
        elif string == 'invroot':
            func = lambda i : 1./np.sqrt(i+1)
        elif string == 'inverse':
            func = lambda i : 1./(i+1)
        elif string == 'invsquare':
            func = lambda i : 1./((i+1)**2)
        else:
            print('Entered a wrong string for the condition of matrix eigenvalues. Please recheck. Resorting to Identitiy')
            func = lambda i : 1
        return func

    def get_scaling_matrix(self, dim_vec):
        P = np.identity(dim_vec)
        for i in range(dim_vec):
            P[i,i] = self.conditon_function(i)
        return P

    def sample_constants(self, dim_vec):

        zeta = np.random.uniform(0., self.upper)
        beta = (self.sigma*np.random.randn(dim_vec) + np.array([self.mean]*dim_vec))
        constant = np.array([self.constant]*dim_vec)
        return zeta, beta, constant

    def function_update(self, x):

        zeta, beta, constant = self.sample_constants(len(x))
        P = self.get_scaling_matrix(dim_vec=len(x))
        Px = np.dot(P, x-constant)
        xPx = np.dot(x-constant, Px)
        Bx = np.dot(beta, x)

        # print(zeta, xPx, Bx)

        return zeta*xPx + Bx

    def grad_update(self, x):

        zeta, beta, constant = self.sample_constants(len(x))
        P = self.get_scaling_matrix(dim_vec=len(x))
        Px =  np.dot(P, x-constant)
        return zeta*2*Px + beta

    def hess_update(self, x):

        pass

class BoothFunction(object):

    def __init__(self):

        self.f_opt = 0
        self.x_opt = np.array([1,3])
        self.a1 = np.array([1,2])
        self.a2 = np.array([2,1])
        self.b1 = 7
        self.b2 = 5

    def function_update(self, x):

        return (np.dot(x, self.a1) - self.b1)**2 + (np.dot(x, self.a2) - self.b2)**2

    def grad_update(self, x):

        return (np.dot(x, self.a1) - self.b1)*self.a2 + (np.dot(x, self.a2) - self.b2)*self.a2

    def hess_update(self, x):

        return np.outer(self.a1, self.a1) + np.outer(self.a2, self.a2)

class BukinN6Function(object):

    def __init__(self, factor = 0.01, constant = 10):
        self.f_opt = 0
        self.x_opt = np.array([-10, 1])
        self.factor = factor
        self.constant = constant

    def function_update(self, x):

        return (1./self.factor)*np.sqrt(np.abs(x[1] - self.factor*x[0]*x[0])) +self.factor*np.abs(x[0]+self.constant)

    def grad_update(self, x):

        g_0 = 50./np.sqrt(np.abs(x[1] - self.factor*x[0]*x[0]))*np.sign(x[1] - self.factor*x[0]*x[0])*(-self.factor*2*x[0]) + self.factor*np.sign(x[0] + self.constant)
        g_1 = 50./np.sqrt(np.abs(x[1] - self.factor*x[0]*x[0]))*np.sign(x[1] - self.factor*x[0]*x[0])

        return np.array([g_0, g_1])

    def hess_update(self, x):

        pass

class MatyasFunction(object):

    def __init__(self):
        self.f_opt = 0
        self.x_opt = np.array([0,0])
        self.c1 = 0.26
        self.c2 = -0.48

    def function_update(self, x):

        return self.c1*(x[0]**2 + x[1]**2) +self.c2*x[0]*x[1]

    def grad_update(self, x):

        g_0 = self.c1*2*x[0] + self.c2*x[1]
        g_1 = self.c1*2*x[1] + self.c2*x[0]

        return np.array([g_0, g_1])

    def hess_update(self, x):

        h_00 = self.c1*2
        h_01 = self.c2
        h_10 = self.c2
        h_11 = self.c1*2

        return np.array([[h_00, h_01], [h_10, h_11]])

class ThreeHumpCamelFunction(object):

    def __init__(self):
        self.x_opt = np.array([0,0])
        self.f_opt = 0
        self.c1 = 2
        self.c2 = -1.05
        self.c3 = 1./6

    def function_update(self, x):

        return self.c1*x[0]**2 + self.c2*x[0]**4 +self.c3*x[0]**6 +x[0]*x[1] +x[1]**2

    def grad_update(self, x):

        g_0 = 2*self.c1*x[0] + 4*self.c2*x[0]**3 +6*self.c3*x[0]**5 + x[1]
        g_1 = x[0] + 2*x[1]

        return np.array([g_0, g_1])

    def hess_update(self, x):

        h_00 = 2*self.c1+ 12*self.c2*x[0]**2 +30*self.c3*x[0]**4
        h_01 = 1
        h_10 = 1
        h_11 = 2

        return np.array([[h_00, h_01], [h_10, h_11]])

class TridFunction(object):
    def __init__(self, dim_vec):
        self.f_opt = -(dim_vec*(dim_vec+4)*(dim_vec-1))/4
        self.x_opt = np.array([i*(dim_vec + 1-i) for i in range(dim_vec)])
        self.c = 1.

    def function_update(self, x):

        return np.dot(x - self.c, x - self.c) - np.dot(x[:-1], x[1:])

    def grad_update(self, x):

        g = np.array([0.]*len(x))
        for i in range(len(x)):
            if i==0:
                g[i] = 2 * (x[i] - self.c) - x[i + 1]
            elif i==len(x)-1:
                g[i] = 2 * (x[i] - self.c) - x[i - 1]
            else:
                g[i] = 2 * (x[i] - self.c) - x[i-1] - x[i+1]
        return g

    def hess_update(self, x):

        pass
