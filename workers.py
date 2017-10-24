from complex import DataComplex


class SimplexWorker:
    # Class to calculate local homology of a simplex (I will use only for edges and vertices)
    workers = []
    epsilon = 0
    dimension = 0
    vertex_homologies = {}

    @classmethod
    def set_params(cls, *, epsilon, dimension):
        cls.epsilon = epsilon
        cls.dimension = dimension

    def __init__(self, simplex, distance_matrix, *, simplex_dim):
        if not SimplexWorker.epsilon or not SimplexWorker.dimension:
            raise ValueError('both epsilon and dimension have to be non-zero, please set the params!')

        if simplex in SimplexWorker.workers:
            raise ValueError("worker for vertex {0} already created!".format(simplex))
        SimplexWorker.workers.append(simplex)

        self.data_complex = DataComplex(distance_matrix=distance_matrix,
                                        dim=SimplexWorker.dimension,
                                        epsilon=SimplexWorker.epsilon)
        self.simplex = simplex
        self.active = False
        # 0-simplex is vertex; 1-simplex is edge
        self.smp_dim = simplex_dim

    def start_calculation(self):
        # potentially the worker could create a thread and here is where parallelism would start
        self.active = True
        self.data_complex.build_vr_complex()
        simplex = set(range(self.smp_dim + 1))
        betti_numbers = self.data_complex.get_localhom(simplex)
        SimplexWorker.vertex_homologies[self.simplex] = betti_numbers
        self.active = False
