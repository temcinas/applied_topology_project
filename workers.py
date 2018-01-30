from complex import VietorisRipsComplex
import multiprocessing as mp
import numpy as np
import math


class VertexWorker:
    # Class to calculate local homology of a vertex and edges coming from it
    manager = mp.Manager()
    workers = []
    epsilon = 0
    space_dimension = 0
    vertex_homologies = manager.dict()
    edge_homologies = manager.dict()
    vertices = []
    dist_to_centers = {}
    dist = None

    @classmethod
    def set_params(cls, *, epsilon, dimension, vertices, dist_funct):
        cls.epsilon = epsilon
        cls.space_dimension = dimension
        cls.vertices = vertices
        cls.dist = dist_funct

    @classmethod
    def clear_params(cls):
        cls.manager = mp.Manager()
        cls.workers = []
        cls.epsilon = 0
        cls.space_dimension = 0
        cls.vertex_homologies = cls.manager.dict()
        cls.edge_homologies = cls.manager.dict()
        cls.vertices = []
        cls.dist_to_centers = {}

    @classmethod
    def _get_closest_center(cls, vertex_index):
        smallest_distance_to_center = math.inf
        relevant_center_id = None

        for center_id, distances in cls.dist_to_centers.items():
            vertex_position = np.argwhere(distances[:, 1:] == vertex_index).flatten()[0]
            distance_to_center = distances[vertex_position][0]
            if distance_to_center < smallest_distance_to_center:
                smallest_distance_to_center = distance_to_center
                relevant_center_id = center_id
        return smallest_distance_to_center, relevant_center_id

    @classmethod
    def get_neighbours(cls, vertex_index):
        vertex = cls.vertices[vertex_index]
        smallest_distance_to_center, relevant_center_id = cls._get_closest_center(vertex_index)

        relevant_distances = cls.dist_to_centers[relevant_center_id]
        relevant_vertices = np.argwhere(relevant_distances[:, :1] <= smallest_distance_to_center + cls.epsilon)
        neighbours = [vertex_index]
        for vertex_coord in relevant_vertices:
            potential_nbr_id = int(relevant_distances[vertex_coord[0], 1])
            if cls.dist(vertex, cls.vertices[potential_nbr_id]) <= cls.epsilon and not potential_nbr_id == vertex_index:
                neighbours.append(potential_nbr_id)
        return neighbours

    @classmethod
    def get_distance_matrix(cls, neighbours):
        matrix = []
        for x in neighbours:
            x = cls.vertices[x]
            row = [cls.dist(x, cls.vertices[y]) for y in neighbours]
            matrix.append(row)
        return np.array(matrix)

    def __init__(self, vertex_id):
        print(vertex_id)
        # Distance_matrix - matrix of pairwise distances
        # Neighbours - ids of neighbours, 0th position is the vertex itself

        if not VertexWorker.epsilon or not VertexWorker.space_dimension:
            raise ValueError('both epsilon and dimension have to be non-zero, please set the params!')

        if vertex_id in VertexWorker.workers:
            raise ValueError("worker for vertex {0} already created!".format(vertex_id))
        VertexWorker.workers.append(vertex_id)

        self.vertex_id = vertex_id
        neighbours = VertexWorker.get_neighbours(vertex_id)
        self.neighbours = neighbours[1:]
        self.distance_matrix = VertexWorker.get_distance_matrix(neighbours)

    def start_calculation(self):
        # potentially the worker could create a thread and here is where parallelism would start
        data_complex = VietorisRipsComplex(distance_matrix=self.distance_matrix,
                                           dim=VertexWorker.space_dimension,
                                           epsilon=VertexWorker.epsilon)
        data_complex.build_vr_complex()
        vertex_homology = data_complex.get_localhom({0})
        VertexWorker.vertex_homologies[self.vertex_id] = vertex_homology
        for neighbour_index, neighbour in enumerate(self.neighbours, 1):
            if self.vertex_id < neighbour:
                edge = frozenset({self.vertex_id, neighbour})
                edge_homology = data_complex.get_localhom({0, neighbour_index})
                VertexWorker.edge_homologies[edge] = edge_homology
