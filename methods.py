import numpy as np
import scipy.optimize as opt
import math

def get_descent_method(descent_type, dim, nodes=1, x=None):

    if descent_type=='grad':
        met = GradDescent(dim=dim, x=x)
    elif descent_type=='test':
        met = TestNewGradDescent(dim=dim, x=x)
    elif descent_type=='accelerated':
        met = AcceleratedDescent(dim=dim, x=x)
    elif descent_type == 'ngd':
        met = NoisyGradDescent(dim=dim, x=x)
    elif descent_type == 'gradtent':
        met = GradTent2Descent(dim=dim, x=x)
    elif descent_type=='primal':
        met = WilburPurePrimalMethod(dim=dim, nodes=nodes)
    elif descent_type=='dladmm':
        met = DLADMMMethod(dim=dim, nodes=nodes)
    else:
        print('You entered some wierd "type". Please correct it. Exiting...')
        exit()
    return met

class WilburPurePrimalMethod(object):
    def __init__(self, dim, nodes):
        self.dim = dim
        self.nodes = nodes
        self.x = np.array([np.zeros(dim)] * nodes)

    def update(self, grad_f, grad_gz, grad_gy, edge_matrix, step_size, active_nodes):
        grad_g = grad_gz
        x_inter = self.x
        for i in active_nodes:
            self_communication_cost = 0
            for j in range(len(edge_matrix[i])):
                self_communication_cost += grad_g(self.x[i], self.x[j])*edge_matrix[i,j]
            self.x[i] = self.x[i] - step_size*(grad_f(self.x[i]) + self_communication_cost)
        return 1

class DLADMMMethod(object):
    # Needs reforms, not working as expected
    def __init__(self, dim, nodes, c=3., rho=50.):
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
        for s_index in range(matrix.shape[0]):
            sum_vec += matrix[s_index, index] * neighbours[s_index]
        return sum_vec

    def update(self, grad_f, grad_gy, grad_gz, edge_matrix, step_size, active_nodes):
        x_inter = self.x
        y_inter = self.y
        z_inter = self.z
        temp = 1
        # print(self.y)
        # print(active_nodes)
        for i in active_nodes:
            neighbours = [j for j in range(edge_matrix.shape[0]) if edge_matrix[i, j] == 1]

            # sum_vec_z = self.summing_vectors(self.z, i, edge_matrix[i])
            sum_vec_z = sum([self.z[j,i] for j in neighbours])
            sum_vec_mu = sum([self.mu[j,i] for j in neighbours])
            # sum_vec_mu = self.summing_vectors(self.mu, i, edge_matrix[i])
            # print(sum(edge_matrix[i]), edge_matrix[i])
            # exit()
            self.x[i] = (1. / (self.c + self.rho * (1 + sum(edge_matrix[i])))) * (-grad_f(self.x[i]) + self.c * self.x[i] - self.lam[i] - sum_vec_mu \
                                                                       + self.rho * self.y[i] + self.rho * sum_vec_z)
            y_inter[i] = (1./(self.c + self.rho))*(sum([-(self.y[i]- self.z[i,j]) for j in neighbours]) + self.c*self.y[i] + self.lam[i] + self.rho*self.x[i])

            # print(neighbours)
            for j in neighbours:
                z_inter[i,j] = (1./(self.c+self.rho))*(-(self.z[i,j]- self.y[i]) + self.c*self.z[i,j] + self.mu[i,j] + self.rho*self.x[j])

            self.y[i] = y_inter[i]
            for j in neighbours:
                self.z[i,j] = z_inter[i,j]

            self.lam[i] = self.lam[i] +self.rho*(self.x[i] - self.y[i])
            for j in range(self.nodes):
                self.mu[i,j] = self.mu[i,j] + self.rho*(self.x[j] - self.z[i,j])

        # print(self.y)
        # exit()

            # print("X ineter :", x_inter[i], " old x  : ", self.x[i])
            # self.x[i] = self.x[i] + step_size*x_inter[i]
            # self.x[i] = (1-step_size)*self.x[i] + step_size*x_inter[i]
            # self.x[i] = x_inter[i]
            # if sum(abs(x_inter[i] - self.x[i])) != 0:
            #     temp = temp*0
        return temp

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
        if x.all() == None:
            self.x = np.random.rand(dim)
        else:
            self.x = x

    def update(self, grad_f, step_size):
        self.x = self.x - step_size*grad_f(self.x)


class NoisyGradDescent(object):
    def __init__(self, dim , x=None):
        self.dim = dim
        if x.all() == None:
            self.x = np.random.rand(dim)
        else:
            self.x = x

    def samplenoise(self):
        a = 0.
        if np.random.rand() < 0.5:
            a = -0.25
        else:
            a = 0.25
        return a

    def update(self, grad_f, step_size):
        eps = self.samplenoise()
        self.x = self.x + np.array([eps for i in range(self.dim)])
        self.x = self.x - step_size*grad_f(self.x)

class GradTent2Descent(object):
    def __init__(self, dim , x=None):
        self.dim = dim
        if x.all() == None:
            self.x = np.random.rand(dim)
        else:
            self.x = x

    def samplenoise(self, const = 0.25):
        a = 0.
        if np.random.rand() < 0.5:
            a = -const
        else:
            a = const
        return a

    def update(self, grad_f, step_size):
        const = 0.25
        grad_value = 0.
        while grad_value < 0.000005 :
            print(const)
            eps = self.samplenoise(const = const)
            print(eps)
            self.y1 = self.x + np.array([eps for i in range(self.dim)])
            self.y2 = self.x - np.array([eps for i in range(self.dim)])
            self.x = self.x - step_size/2.*grad_f(self.y1) - step_size/2.*grad_f(self.y2)
            grad_value = np.linalg.norm(grad_f(self.y1) + grad_f(self.y2))
            print(grad_f(self.y1), grad_f(self.y1), grad_value)
            print("Const before :", const)
            const = const*2
            print("Const after :", const)


class GradTentDescent(object):
    def __init__(self, dim , x=None):
        self.dim = dim
        if x.all() == None:
            self.x = np.random.rand(dim)
        else:
            self.x = x

    def samplenoise(self):
        a = 0.
        if np.random.rand() < 0.5:
            a = -0.25
        else:
            a = 0.25
        return a

    def update(self, grad_f, step_size):
        eps = self.samplenoise()
        self.y1 = self.x + np.array([eps for i in range(self.dim)])
        self.y2 = self.x - np.array([eps for i in range(self.dim)])
        self.x = self.x - step_size/2.*grad_f(self.y1) - step_size/2.*grad_f(self.y2)


class TestNewGradDescent(object):
    def __init__(self, dim , x=None):
        self.dim = dim
        if x==None:
            self.x = np.random.rand(dim)
        else:
            self.x = x

    def update(self, grad_f, step_size):
        self.x = self.x - 2*step_size*grad_f(self.x)*self.x + (step_size*grad_f(self.x))**2*self.x


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