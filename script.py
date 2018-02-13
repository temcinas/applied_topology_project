import numpy as np
import pandas as pd
from manager import DatasetManager
from notebook_helpers import get_word_clusters
import math
import scipy.spatial.distance as dist
import pickle


def produce_data(df, vertices, distance_funct, start, end, step):
    c_list = []
    for epsilon in np.arange(start, end, step):
        manager = DatasetManager(vertices=vertices,
                                 centers_num=lambda x: int(math.sqrt(x)),
                                 distance_funct=distance_funct,
                                 epsilon=epsilon)
        manager.get_centers_ready()
        manager.calculate_homologies()
        with open('out_clustering.txt', 'a') as fout:
            fout.write('Current epsilon: {0}'.format(epsilon))
        manager.cluster(report_homologies=True)
        clusters = get_word_clusters(df, manager.clusters)
        c_list.append(clusters)
    return c_list


def arccosdist(vect1, vect2):
    if (vect1 == vect2).all():
        return 0
    return math.degrees(np.arccos(1 - dist.cosine(vect1, vect2)))


def execute_everything(start, end, step):
    df = pd.read_csv('words_df.csv')
    df[0] = df["0"]
    del df['Unnamed: 0'], df["0"]

    vertices = np.loadtxt('glove_data.txt')
    cluster_list = produce_data(df, vertices=vertices, distance_funct=arccosdist,
                                start=start, end=end, step=step)

    with open('cluster_list_glove_{0}_{1}_{2}.pkl'.format(start, end, step), 'wb') as f:
        pickle.dump(cluster_list, f)

    return


if __name__ == "__main__":
    execute_everything(start=40, end=76, step=1)
