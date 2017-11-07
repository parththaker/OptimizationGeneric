import numpy as np
import scipy.optimize as opt
import math

def get_descent_method(descent_type, dim, x=None):

    if descent_type=='grad':
        met = GradDescent(dim=dim, x=x)
    elif descent_type=='accelerated':
        met = AcceleratedDescent(dim=dim, x=x)
    elif descent_type=='stoc':
        met = StochasticMethod1(dim=dim, x=x)
    else:
        print('You entered some wierd "type". Please correct it. Exiting...')
        exit()
    return met

class DLADMMMethod(object):
    # Needs reforms, not working as expected
    def __init__(self, dim, nodes, c, rho):
        self.c = c
        self.dim = dim
        self.nodes = nodes
        self.rho = rho

        self.x = np.array([np.zeros(dim)] * nodes)
        self.y = np.array([np.zeros(dim)] * nodes)
        self.z = np.array([[np.zeros(dim)] * nodes] * nodes)
        self.lam = np.array([np.zeros(dim)] * nodes)
        self.mu = np.array([[np.zeros(dim)] * nodes] * nodes)

    def summing_vectors(self, matrix, index, neighbours):
        sum_vec = np.zeros(matrix.shape[-1])
        for i in range(matrix.shape[0]):
            sum_vec += matrix[i, index] * neighbours[i]
        return sum_vec

    def update(self, grad_f, grad_gy, grad_gz, edge_matrix, step_size, active_nodes):
        x_inter = self.x
        for i in active_nodes:
            sum_vec_z = self.summing_vectors(self.z, i, edge_matrix[i])
            sum_vec_mu = self.summing_vectors(self.mu, i, edge_matrix[i])
            x_inter[i] = (1. / (self.c + self.rho * (1 + sum(edge_matrix[i])))) * (-grad_f(self.x[i]) + self.c * self.x[i] + -self.lam[i] - sum_vec_mu \
                                                                       + self.rho * self.y[i] + self.rho * sum_vec_z)
            self.y[i] = (1./(self.c + self.rho))*(sum([-grad_gy(self.y[i], self.z[i,j]) for j in range(self.nodes)]) + self.c*self.y[i] + self.lam[i] + self.rho*x_inter[i])
            self.lam[i] = self.lam[i] +self.rho*(x_inter[i] - self.y[i])

            neighbours = [j for j in range(edge_matrix.shape[0]) if edge_matrix[i, j] == 1]
            for j in neighbours:
                self.z[i,j] = (1./(self.c+self.rho))*(-grad_gz(self.z[i,j], self.y[i]) + self.c*self.z[i,j] + self.mu[i,j] + self.rho*x_inter[j])
                self.mu[i,j] = self.mu[i,j] + self.rho*(x_inter[j] - self.z[i,j])
            print("X ineter :", x_inter[i], " old x  : ", self.x[i])
            self.x[i] = self.x[i] + step_size*x_inter[i]

class DADMMMethod(object):
    # Doesnt work needs reformation
    def __init__(self, dim, nodes, c, rho):
        self.c = c
        self.dim = dim
        self.nodes = nodes
        self.rho = rho

        self.x = np.array([np.zeros(dim)] * nodes)
        self.y = np.array([np.zeros(dim)] * nodes)
        self.z = np.array([[np.zeros(dim)] * nodes] * nodes)
        self.lam = np.array([np.zeros(dim)] * nodes)
        self.mu = np.array([[np.zeros(dim)] * nodes] * nodes)

    def optimize_x(self, x, f, edge_matrix, current_node):
        x = x.reshape((10,3))
        neighbours = [j for j in range(edge_matrix.shape[0]) if edge_matrix[current_node, j] == 1]
        l2_dist_z = sum([np.linalg.norm(x[neigh] - self.z[current_node, neigh])**2 for neigh in neighbours])
        l2_dist_y = np.linalg.norm(x[current_node] - self.y[current_node])**2
        dist_lam = np.dot(self.lam[current_node], x[current_node])
        dist_mu = sum([np.dot(self.mu[current_node, neigh], x[neigh]) for neigh in neighbours])
        cost = f(x[current_node]) + dist_lam + dist_mu + self.rho/2*(l2_dist_y + l2_dist_z)
        return cost

    def optimize_y(self, y, g, edge_matrix, current_node):
        y = y.reshape((10,3))
        neighbours = [j for j in range(edge_matrix.shape[0]) if edge_matrix[current_node, j] == 1]
        l2_dist_z = sum([np.linalg.norm(self.x[neigh] - self.z[current_node, neigh])**2 for neigh in neighbours])
        l2_dist_y = np.linalg.norm(self.x[current_node] - y[current_node])**2
        dist_lam = np.dot(self.lam[current_node], y[current_node])
        dist_mu = sum([np.dot(self.mu[current_node, neigh], self.z[current_node, neigh]) for neigh in neighbours])
        l2_dist_zy = sum([g(self.z[current_node, neigh], y[current_node]) for neigh in neighbours])
        cost = l2_dist_zy - dist_lam - dist_mu + self.rho/2.*(l2_dist_y + l2_dist_z)
        return cost

    def optimize_z(self, z, g, edge_matrix, current_node):
        z = z.reshape((10,10,3))
        neighbours = [j for j in range(edge_matrix.shape[0]) if edge_matrix[current_node, j] == 1]
        l2_dist_z = sum([np.linalg.norm(self.x[neigh] - z[current_node, neigh])**2 for neigh in neighbours])
        l2_dist_y = np.linalg.norm(self.x[current_node] - self.y[current_node])**2
        dist_lam = np.dot(self.lam[current_node], self.y[current_node])
        dist_mu = sum([np.dot(self.mu[current_node, neigh], z[current_node, neigh]) for neigh in neighbours])
        l2_dist_zy = sum([g(z[current_node, neigh], self.y[current_node]) for neigh in neighbours])
        cost = l2_dist_zy - dist_lam - dist_mu + self.rho/2.*(l2_dist_y + l2_dist_z)
        return cost

    def update(self, f, g, edge_matrix, step_size, active_nodes):
        for i in active_nodes:
            print("Active node no. :", i)
            opti_x = opt.minimize(self.optimize_x, x0=self.x, args=(f, edge_matrix, i, ))
            print("X error : ", opti_x.fun)
            new_x = opti_x.x
            new_x = new_x.reshape((10,3))
            opti_y = opt.minimize(self.optimize_y, x0 = self.y, args=(g, edge_matrix, i, ))
            print("Y error : ", opti_y.fun)
            new_y = opti_y.x
            new_y = new_y.reshape((10,3))
            opti_z  = opt.minimize(self.optimize_z, x0=self.z, args = (g, edge_matrix, i, ))
            print("Z error : ", opti_z.fun)
            new_z = opti_z.x
            new_z = new_z.reshape((10, 10,3))

            self.x = self.x + step_size*new_x
            print(self.x)
            self.y = self.y + step_size*new_y
            self.z = self.z + step_size * new_z

            self.lam[i] = self.lam[i] +self.rho*(self.x[i] - self.y[i])
            neighbours = [j for j in range(edge_matrix.shape[0]) if edge_matrix[i, j] == 1]
            for j in neighbours:
                self.mu[i,j] = self.mu[i,j] + self.rho*(self.x[j] - self.z[i,j])


class GradDescent(object):
    def __init__(self, dim , x=None):
        self.dim = dim
        if x==None:
            self.x = np.random.rand(dim)
        else:
            self.x = x

    def update(self, grad_f, step_size):
        self.x = self.x - step_size*grad_f(self.x)

class AcceleratedDescent(object):
    def __init__(self, dim, x=None):

        self.lam_current = 0.
        self.lam_old = 0.
        self.gamma = 0.
        if x==None:
            self.x = np.random.rand(dim)
        else:
            self.x = x
        self.y = self.x

    def lambda_update(self):
        self.lam_old = self.lam_current
        self.lam_current = (1+math.sqrt(1+4*self.lam_old**2))/2.

    def gamma_update(self):
        self.gamma = (1-self.lam_old)/self.lam_current

    def update(self, grad_f, step_size):
        self.lambda_update()
        self.gamma_update()
        new_y = self.x - step_size*grad_f(self.x)
        self.x = (1-self.gamma)*new_y + self.gamma*self.y
        self.y = new_y


class StochasticMethod1(object):
    def __init__(self, dim, x=None):
        self.lam = 1.
        self.gamma = 1.
        if x==None:
            self.x = np.random.randn(dim)
        else:
            self.x = x
        self.y = self.x
        self.count = 1.

    def gamma_update(self):
        self.gamma = self.gamma + self.lam

    def lambda_update(self):
        self.lam = np.sqrt(self.count + 1.)

    def update(self, grad_f, step_size):
        self.y = self.x - step_size*grad_f(self.x)
        self.x = (1-self.lam/self.gamma)*self.x + (self.lam/self.gamma)*self.y
        # print((1-self.lam/self.gamma), (self.lam/self.gamma), self.count)
        self.count = self.count+1
        self.lambda_update()
        self.gamma_update()