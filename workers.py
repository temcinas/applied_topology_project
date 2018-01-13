from complex import DataComplex


class SimplexWorker:
    # Class to calculate local homology of a simplex (I will use only for edges and vertices)
    workers = []
    epsilon = 0
    space_dimension = 0
    vertex_homologies = {}

    @classmethod
    def set_params(cls, *, epsilon, dimension):
        cls.epsilon = epsilon
        cls.space_dimension = dimension

    def __init__(self, simplex, distance_matrix, *, simplex_dim, neighbours=None):
        # Simplex - either vertex_id or a 2-element set of vertex ids
        # Distance_matrix - matrix of pairwise distances
        # Neighbours - ids of neighbours, if applicable, 0th position is the vertex itself

        if not SimplexWorker.epsilon or not SimplexWorker.space_dimension:
            raise ValueError('both epsilon and dimension have to be non-zero, please set the params!')

        if simplex in SimplexWorker.workers:
            raise ValueError("worker for vertex {0} already created!".format(simplex))
        SimplexWorker.workers.append(simplex)

        self.distance_matrix = distance_matrix
        self.simplex = simplex
        self.active = False
        # 0-simplex is vertex; 1-simplex is edge
        self.smp_dim = simplex_dim
        if neighbours:
            self.neighbours = neighbours

    def start_calculation(self):
        # potentially the worker could create a thread and here is where parallelism would start
        self.active = True
        data_complex = DataComplex(distance_matrix=self.distance_matrix,
                                   dim=SimplexWorker.space_dimension,
                                   epsilon=SimplexWorker.epsilon)
        data_complex.build_vr_complex()
        # TODO: I ended here
        simplex = set(range(self.smp_dim + 1))
        betti_numbers = data_complex.get_localhom(simplex)
        SimplexWorker.vertex_homologies[self.simplex] = betti_numbers
        self.active = False
