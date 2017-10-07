import numpy as np

def neighbour(matrix, index):
    if matrix.shape[0]!=matrix.shape[1]:
        print("Please give a square matrix")
        return -1
    else:
        return matrix[index]

class GenerateGraph(object):
    @staticmethod
    def generate_graph(nodes, type, prob=0.5, star_loc = 0):
        base_matrix = np.random.rand(nodes, nodes)
        if type=='random':
            graph = GenerateGraph.create_random_graph(base_matrix, prob=prob)
        elif type=='complete':
            graph = GenerateGraph.create_complete_graph(base_matrix)
        elif type=='line':
            graph = GenerateGraph.create_line_graph(base_matrix)
        elif type=='star':
            graph = GenerateGraph.create_star_graph(base_matrix, star_loc=star_loc)
        else:
            print("You have entered incorrect type. Generating random matrix.")
            graph = base_matrix
        return graph

    @staticmethod
    def create_random_graph(edge_matrix, prob):
        # This is Erdos-Renyi model graph generator function. Each node has the probability of being present with probability p.
        nodes = edge_matrix.shape[0]
        for i in range(nodes):
            for j in range(i + 1):
                if i == j:
                    edge_matrix[i, j] = 0
                elif edge_matrix[i, j] + edge_matrix[j, i] < 2*prob:
                    edge_matrix[i, j] = 1
                    edge_matrix[j, i] = 1
                else:
                    edge_matrix[i, j] = 0
                    edge_matrix[j, i] = 0
        return edge_matrix

    @staticmethod
    def create_complete_graph(edge_matrix):
        nodes = edge_matrix.shape[0]
        for i in range(nodes):
            for j in range(nodes):
                if i == j:
                    edge_matrix[i, j] = 0
                else:
                    edge_matrix[j, i] = 1
        return edge_matrix

    @staticmethod
    def create_line_graph(edge_matrix):
        nodes = edge_matrix.shape[0]
        for i in range(nodes):
            for j in range(nodes):
                if abs(i - j)==1 or abs(i-j-nodes)==1 or abs(i-j+nodes)==1:
                    edge_matrix[i, j] = 1
                    edge_matrix[j, i] = 1
                else:
                    edge_matrix[j, i] = 0
        return edge_matrix

    @staticmethod
    def create_star_graph(edge_matrix, star_loc = 0):
        nodes = edge_matrix.shape[0]
        for i in range(nodes):
            for j in range(nodes):
                if(j==star_loc or i==star_loc )and i!=j :
                    edge_matrix[i, j] = 1
                    edge_matrix[j, i] = 1
                else:
                    edge_matrix[j, i] = 0
        return edge_matrix
