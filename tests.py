import unittest
import numpy as np
import math

from snf import get_arg_absmin, put_in_snf, get_snf
from manager import DatasetManager


class TestSnf(unittest.TestCase):

    matrix1 = np.array([[25,   -300,    1050,   -1400,    630],
                       [-300,   4800,  -18900,   26880, -12600],
                       [1050, -18900,   79380, -117600,  56700],
                       [-1400,  26880, -117600,  179200, -88200],
                       [630, -12600,   56700,  -88200,  44100]])
    matrix2 = np.array([[5, 0, 0, 0, 0],
                        [0, 60, 0, 0, 0],
                        [0, 0, -18900, -3780, -16380],
                        [0, 0, -10500, -3780, -7140],
                        [0, 0, -12180, -4620, -7980]])

    def test_absmin(self):
        matrix = TestSnf.matrix2.copy()
        row, col = get_arg_absmin(matrix, 2)
        self.assertEqual(row, 3, msg='absmin function in SNF has failed')
        self.assertEqual(col, 3, msg='absmin function in SNF has failed')

    def test_put_snf(self):
        matrix = TestSnf.matrix1.copy()
        put_in_snf(matrix)
        correct_result = np.array([[5,    0,    0,    0,    0],
                                   [0,   60,    0,    0,    0],
                                   [0,    0,  420,    0,    0],
                                   [0,    0,    0,  840,    0],
                                   [0,    0,    0,    0, 2520]])
        self.assertTrue((matrix == correct_result).all(), msg="put_in_snf function in SNF has failed.")

    def test_get_snf(self):
        matrix = TestSnf.matrix1.copy()
        get_snf(matrix)
        correct_result = np.array([[5, 0, 0, 0, 0],
                                   [0, 60, 0, 0, 0],
                                   [0, 0, 420, 0, 0],
                                   [0, 0, 0, 840, 0],
                                   [0, 0, 0, 0, 2520]])
        self.assertTrue((matrix == correct_result).all(), msg="get_snf function in SNF has failed.")


class TestHomology(unittest.TestCase):
    points1 = [np.array([0, 0, 1, 0]), np.array([1, 0, 1, 0]), np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0])]
    points2 = [np.array([0, 0, 0]), np.array([0, 1, 0]), np.array([0, 1/2, math.sqrt(3)/2]),
               np.array([-math.sqrt(3)/2, 1/2, 0]), np.array([0, 1/2, -math.sqrt(3)/2])]

    def test_terahedron(self):
        manager = DatasetManager(vertex_iter=iter(TestHomology.points1),
                                 centers_num=lambda x: int(math.sqrt(x)),
                                 distance_funct=lambda x, y: np.linalg.norm(x-y),
                                 epsilon=3)
        manager.get_centers_ready()
        worker = manager.calulate_homologies()

        for homology in worker.vertex_homologies.values():
            self.assertEqual(homology, [0, 0, 0, 0], msg='tetrahedron vertex homology is not correct')

        for homology in worker.edge_homologies.values():
            self.assertEqual(homology, [0, 0, 0, 0], msg='tetrahedron edge homology is not correct')

    def test_three_sheets(self):
        manager = DatasetManager(vertex_iter=iter(TestHomology.points2),
                                 centers_num=lambda x: int(math.sqrt(x)),
                                 distance_funct=lambda x, y: np.linalg.norm(x - y),
                                 epsilon=1)
        manager.get_centers_ready()
        worker = manager.calulate_homologies()

        for homology in worker.vertex_homologies.values():
            self.assertEqual(homology, [0, 0, 0], msg='3-sheet vertex homology is not correct')

        special = frozenset({0, 1})
        for edge, homology in worker.edge_homologies.items():
            if not edge == special:
                self.assertEqual(homology, [0, 0, 0], msg='3-sheet edge homology is not correct')
        self.assertEqual(worker.edge_homologies[special], [0, 0, 2], msg='3-sheet edge homology is not correct')

if __name__ == '__main__':
    unittest.main()
