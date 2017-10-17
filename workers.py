from complex import DataComplex


class SimplexWorker:
    # maybe this should be a factory rather than a class?
    # then we could have different parameters (epsilon, dim) for different simplices
    # also, maybe then we would not need to pass them to init?
    workers = []
    epsilon = 0
    dimension = 0
    parameters_are_set = False
    vertex_homologies = {}

    def __init__(self, simplex, distance_matrix, *, simplex_dim, epsilon, dimension):
        if not SimplexWorker.parameters_are_set:
            SimplexWorker.epsilon = epsilon
            SimplexWorker.dimension = dimension
            SimplexWorker.parameters_are_set = True

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
        # potentially the worker could create a thread and here is where parallelism starts
        self.active = True
        # TODO: shoot off building of VR here rather than in DataComplex init
        simplex = set(range(self.smp_dim + 1))
        betti_numbers = self.data_complex._get_localhom(simplex)
        SimplexWorker.vertex_homologies[self.simplex] = betti_numbers
        self.active = False
