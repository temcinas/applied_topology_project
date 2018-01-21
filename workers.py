from complex import VietorisRipsComplex
from datetime import datetime


class VertexWorker:
    # Class to calculate local homology of a vertex and edges coming from it
    workers = []
    epsilon = 0
    space_dimension = 0
    vertex_homologies = {}
    edge_homologies = {}

    @classmethod
    def set_params(cls, *, epsilon, dimension):
        cls.epsilon = epsilon
        cls.space_dimension = dimension

    @classmethod
    def clear_params(cls):
        cls.workers = []
        cls.epsilon = 0
        cls.space_dimension = 0
        cls.vertex_homologies = {}
        cls.edge_homologies = {}

    def __init__(self, vertex_id, distance_matrix, neighbours):
        print('Vertex_id: {0} , time: {1}'.format(vertex_id, datetime.now()))
        # Distance_matrix - matrix of pairwise distances
        # Neighbours - ids of neighbours, 0th position is the vertex itself

        if not VertexWorker.epsilon or not VertexWorker.space_dimension:
            raise ValueError('both epsilon and dimension have to be non-zero, please set the params!')

        if vertex_id in VertexWorker.workers:
            raise ValueError("worker for vertex {0} already created!".format(vertex_id))
        VertexWorker.workers.append(vertex_id)

        self.distance_matrix = distance_matrix
        self.vertex_id = vertex_id
        self.active = False
        self.neighbours = neighbours[1:]

    def start_calculation(self):
        # potentially the worker could create a thread and here is where parallelism would start
        self.active = True
        data_complex = VietorisRipsComplex(distance_matrix=self.distance_matrix,
                                           dim=VertexWorker.space_dimension,
                                           epsilon=VertexWorker.epsilon)
        data_complex.build_vr_complex()
        # print('Starting homology calculation of the vertex {0}'.format(datetime.now()))
        vertex_homology = data_complex.get_localhom({0})
        VertexWorker.vertex_homologies[self.vertex_id] = vertex_homology
        # print(self.neighbours)
        for neighbour_index, neighbour in enumerate(self.neighbours, 1):
            edge = frozenset({self.vertex_id, neighbour})
            if edge not in VertexWorker.edge_homologies.keys():
                # print('Starting homology calculation of the edge {0} {1}'.format(edge, datetime.now()))
                edge_homology = data_complex.get_localhom({0, neighbour_index})
                VertexWorker.edge_homologies[edge] = edge_homology
        self.active = False
