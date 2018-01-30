import numpy as np
import multiprocessing as mp

from random import sample
from collections import defaultdict, Counter

from workers import VertexWorker
from complex import VietorisRipsComplex
from helpers import get_boundary_operator


class DatasetManager:
    def __init__(self, vertices, centers_num, distance_funct, epsilon, space_dimension=None, n_processes=4):
        # entry point of the API
        # vertex_iter - iterator of vertices themselves (i.e. vectors)
        # centers_num - function taking int (no of vertices) and returning an int - number of centers
        # distance_funct - a function that takes two vectors and returns distance between them
        # epsilon - threshold parameter

        self.vertices = np.array(vertices)
        self.distance = distance_funct
        self.clusters = []
        self.n_processes = n_processes

        num_vertices = len(self.vertices)

        self.pool = mp.Queue()
        for i in range(num_vertices):
            self.pool.put(i)

        self.center_indexes = sample(range(num_vertices), centers_num(num_vertices))
        self.dist_to_centers = {}

        self.epsilon = epsilon
        self.space_dimension = space_dimension or len(self.vertices[0])

        VertexWorker.clear_params()
        VertexWorker.set_params(epsilon=self.epsilon,
                                dimension=self.space_dimension,
                                vertices=self.vertices,
                                dist_funct=distance_funct)

    def get_centers_ready(self):
        dist_to_centers = {}
        for center_index in self.center_indexes:
            center = self.vertices[center_index]
            distances = sorted([(self.distance(center, vertex), i) for i, vertex in enumerate(self.vertices)])
            dist_to_centers[center_index] = np.array(distances)
        VertexWorker.dist_to_centers = dist_to_centers

    def process_funct(self):
        while not self.pool.empty():
            vertex_id = self.pool.get()
            worker = VertexWorker(vertex_id=vertex_id)
            worker.start_calculation()

    def calculate_homologies(self):
        processes = []
        for p in range(self.n_processes):
            process = mp.Process(target=self.process_funct)
            process.start()
            processes.append(process)

        for process in processes:
            process.join()
        return VertexWorker

    @staticmethod
    def _get_cluster_adjacency_dict(vertex_homologies, edge_homologies):
        adjacency_dict = defaultdict(list)
        for (vertex1, vertex2), homology in edge_homologies.items():
            if vertex_homologies[vertex1] == homology and vertex_homologies[vertex2] == homology:
                adjacency_dict[vertex1].append(vertex2)
                adjacency_dict[vertex2].append(vertex1)

        for vertex in vertex_homologies.keys():
            if vertex not in adjacency_dict.keys():
                adjacency_dict[vertex] = []
        return adjacency_dict

    def _visit_neighbours(self, vertex, cluster, adjacency_dict, visited):
        neighbours = adjacency_dict[vertex]
        for neighbour in neighbours:
            if neighbour not in visited:
                visited.add(neighbour)
                cluster.append(neighbour)
                self._visit_neighbours(neighbour, cluster, adjacency_dict, visited)

    def cluster(self, report_homologies=False):
        vertex_homologies, edge_homologies = VertexWorker.vertex_homologies, VertexWorker.edge_homologies
        if not vertex_homologies:
            raise ValueError('no homology groups have been calculated, use DatasetManager.calculate_homologies()')

        adjacency_dict = self._get_cluster_adjacency_dict(vertex_homologies, edge_homologies)
        visited = set()
        for vertex, neighbours in adjacency_dict.items():
            if vertex not in visited:
                visited.add(vertex)
                cluster = [vertex]
                self._visit_neighbours(vertex, cluster, adjacency_dict, visited)
                self.clusters.append(cluster)

        if report_homologies:
            new_clusters = []
            for cluster in self.clusters:
                homology = vertex_homologies[cluster[0]]
                new_clusters.append((cluster, homology))
            self.clusters = new_clusters

    def report_on_vertex(self, vertex_id):
        # returns a dict with keys being dimensions of simplices and values - no. of simplices of that dim in local VR
        # also returns a dict with keys being dimensions of operators and values - % of non-zero entries there
        neighbours = VertexWorker.get_neighbours(vertex_id)
        distance_matrix = VertexWorker.get_distance_matrix(neighbours)
        cplx = VietorisRipsComplex(distance_matrix, self.epsilon, self.space_dimension)
        cplx.build_vr_complex()
        local_vr = cplx.get_relevant_subcomplex({0})
        simplices_counter = Counter([len(simplex) - 1 for simplex in local_vr])
        operators = [get_boundary_operator(local_vr, dim) for dim in range(cplx.dim)]
        operators_counter = {i: 100 * len(np.argwhere(operator != 0)) / operator.size
                             for i, operator in enumerate(operators) if operator.size}
        return simplices_counter, operators_counter
