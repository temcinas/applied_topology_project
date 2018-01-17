import numpy as np
import math

from random import sample
from collections import defaultdict

from workers import VertexWorker


class DatasetManager:
    def __init__(self, vertex_iter, centers_num, distance_funct, epsilon, space_dimension=None):
        # entry point of the API
        # vertex_iter - iterator of vertices themselves (i.e. vectors)
        # centers_num - function taking int (no of vertices) and returning an int - number of centers
        # distance_funct - a function that takes two vectors and returns distance between them
        # epsilon - threshold parameter

        self.vertices = np.array(list(vertex_iter))
        self.distance = distance_funct
        self.clusters = []

        num_vertices = len(self.vertices)
        self.center_indexes = sample(range(num_vertices), centers_num(num_vertices))
        self.dist_to_centers = {}

        self.epsilon = epsilon
        self.space_dimension = space_dimension or len(self.vertices[0])

        VertexWorker.clear_params()
        VertexWorker.set_params(epsilon=self.epsilon, dimension=self.space_dimension)

    def get_centers_ready(self):
        for center_index in self.center_indexes:
            center = self.vertices[center_index]
            distances = sorted([(self.distance(center, vertex), i) for i, vertex in enumerate(self.vertices)])
            self.dist_to_centers[center_index] = np.array(distances)

    def _get_closest_center(self, vertex_index):
        smallest_distance_to_center = math.inf
        relevant_center_id = None

        for center_id, distances in self.dist_to_centers.items():
            vertex_position = np.argwhere(distances[:, 1:] == vertex_index).flatten()[0]
            distance_to_center = distances[vertex_position][0]
            if distance_to_center < smallest_distance_to_center:
                smallest_distance_to_center = distance_to_center
                relevant_center_id = center_id
        return smallest_distance_to_center, relevant_center_id

    def _get_neighbours(self, vertex_index):
        vertex = self.vertices[vertex_index]
        smallest_distance_to_center, relevant_center_id = self._get_closest_center(vertex_index)

        relevant_distances = self.dist_to_centers[relevant_center_id]
        relevant_vertices = np.argwhere(relevant_distances[:, :1] <= smallest_distance_to_center + self.epsilon)
        neighbours = [vertex_index]
        for vertex_coord in relevant_vertices:
            potential_nbr_id = int(relevant_distances[vertex_coord[0], 1])
            if self.distance(vertex, self.vertices[potential_nbr_id]) <= self.epsilon and not potential_nbr_id == vertex_index:
                neighbours.append(potential_nbr_id)
        return neighbours

    def _get_distance_matrix(self, neighbours):
        matrix = []
        for x in neighbours:
            x = self.vertices[x]
            row = [self.distance(x, self.vertices[y]) for y in neighbours]
            matrix.append(row)
        return np.array(matrix)

    def _visit_vertex(self, vertex_index, neighbours):
        distance_matrix = self._get_distance_matrix(neighbours)
        worker = VertexWorker(vertex_index, distance_matrix, neighbours)
        worker.start_calculation()

    def calculate_homologies(self):
        for vertex_index, vertex in enumerate(self.vertices):
            neighbours = self._get_neighbours(vertex_index)
            self._visit_vertex(vertex_index, neighbours)
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
        # print(visited)
        neighbours = adjacency_dict[vertex]
        for neighbour in neighbours:
            # if_visited = visited.get(neighbour, False)
            # if not if_visited:
            if neighbour not in visited:
                visited.add(neighbour)
                # visited[neighbour] = True
                cluster.append(neighbour)
                self._visit_neighbours(neighbour, cluster, adjacency_dict, visited)

    def cluster(self, report_homologies=False):
        vertex_homologies, edge_homologies = VertexWorker.vertex_homologies, VertexWorker.edge_homologies
        if not vertex_homologies or not edge_homologies:
            raise ValueError('no homology groups have been calculated, use DatasetManager.calculate_homologies()')

        adjacency_dict = self._get_cluster_adjacency_dict(vertex_homologies, edge_homologies)
        # print(adjacency_dict)
        # visited = {}
        visited = set()
        for vertex, neighbours in adjacency_dict.items():
            # if_visited = visited.get(vertex, False)
            # if not if_visited:
            if vertex not in visited:
                visited.add(vertex)
                # visited[vertex] = True
                cluster = [vertex]
                self._visit_neighbours(vertex, cluster, adjacency_dict, visited)
                self.clusters.append(cluster)

        if report_homologies:
            new_clusters = []
            for cluster in self.clusters:
                homology = vertex_homologies[cluster[0]]
                new_clusters.append((cluster, homology))
            self.clusters = new_clusters
