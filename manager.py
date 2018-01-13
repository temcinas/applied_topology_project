import numpy as np
import math

from random import sample
from collections import deque

from workers import VertexWorker


class DatasetManager:
    def __init__(self, vertex_iter, centers_num, distance_funct, epsilon):
        # entry point of the API
        # vertex_iter - iterator of vertices themselves (i.e. vectors)
        # centers_num - function taking int (no of vertices) and returning an int - number of centers
        # distance_funct - a function that takes two vectors and returns distance between them
        # epsilon - threshold parameter

        self.vertices = np.array(vertex_iter)
        self.distance = distance_funct

        num_vertices = len(self.vertices)
        self.center_indexes = sample(range(num_vertices), centers_num(num_vertices))
        self.dist_to_centers = {}
        self.epsilon = epsilon

    def get_centers_ready(self):
        for center_index in self.center_indexes:
            center = self.vertices[center_index]
            distances = sorted([(self.distance(center, vertex), vertex) for vertex in self.vertices])
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
            potential_nbr_id = relevant_distances[vertex_coord[0], 1]
            if self.distance(vertex, self.vertices[potential_nbr_id]) <= self.epsilon and not potential_nbr_id == vertex_index:
                neighbours.append(potential_nbr_id)

        return neighbours

    def _get_distance_matrix(self, neighbours):
        matrix = []
        for x in neighbours:
            row = [self.distance(x, y) for y in neighbours]
            matrix.append(row)
        return np.array(matrix)

    def _visit_vertex(self, vertex_index, neighbours):
        distance_matrix = self._get_distance_matrix(neighbours)
        worker = VertexWorker(vertex_index, distance_matrix, neighbours)
        worker.start_calculation()

    def calulate_homologies(self):
        for vertex_index, vertex in enumerate(self.vertices):
            neighbours = self._get_neighbours(vertex_index)
            self._visit_vertex(vertex_index, neighbours)

    # def calculate_homologies(self, k=100):
    #     # TODO: think if we need this complicated thing, maybe just go through all vertices and that's it
    #     last_k = deque([])
    #     visited = set()
    #     queue = deque([0])
    #     while queue:
    #         vertex_index = queue.popleft()
    #         if vertex_index not in visited:
    #             visited.add(vertex_index)
    #             neighbours = self._get_neighbours(vertex_index)
    #             self._visit_vertex(vertex_index, neighbours)
    #
    #             if len(last_k) > k:
    #                 last_k.popleft()
    #             last_k.append((vertex_index, neighbours))
    #
    #             queue.extend(neighbours)

    def cluster(self):
        # TODO: implement
        pass
