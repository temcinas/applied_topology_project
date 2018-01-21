from collections import defaultdict
from manager import DatasetManager

import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import math
import numpy as np

GOOD_TYPES = ['word2vec', 'glove']
GOOD_DISTANCES = ['euclidean', 'arccos']


def setup_plotly():
    plotly.tools.set_credentials_file(username='tadas.t', api_key='Ngz5K6kLcZm19NzZxH9b')


def get_word_clusters(df, clusters):
    new_clusters = []
    for cluster in clusters:
        cluster, hom = cluster
        new_cluster = [df[0][x] for x in cluster]
        new_clusters.append((new_cluster, hom[:10]))
    return new_clusters


def get_wordvec(word, model, model_type):
    if model_type not in GOOD_TYPES:
        raise ValueError('bad model type')
    if model_type == 'word2vec':
        return model[word]
    if model_type == 'glove':
        return model(word).vector


def produce_data(df, vertices, distance_funct, start, end, step):
    cluster_list = []
    for epsilon in np.arange(start, end, step):
        manager = DatasetManager(vertices=vertices,
                                 centers_num=lambda x: int(math.sqrt(x)),
                                 distance_funct=distance_funct,
                                 epsilon=epsilon)
        manager.get_centers_ready()
        manager.calculate_homologies()
        print('Started clustering {0}'.format(epsilon))
        manager.cluster(report_homologies=True)
        clusters = get_word_clusters(df, manager.clusters)
        cluster_list.append(clusters)
    return cluster_list


def prepare_data(cluster_list):
    word_y_values = defaultdict(list)
    word_homologies = defaultdict(list)

    for clusters in cluster_list:
        for i, stuff in enumerate(clusters):
            cluster, homology = stuff
            for word in cluster:
                word_y_values[word].append(i)
                word_homologies[word].append(homology)
    return word_y_values, word_homologies


def plot_data(relevant_words, x_values, word_y_values, word_homologies, filename, title):
    data = []
    for word in relevant_words:
        trace = go.Scatter(
            x=x_values,
            y=word_y_values[word],
            text=[str(x) for x in word_homologies[word]],
            name=word,
            hoverinfo='text+name'
        )
        data.append(trace)

    layout = go.Layout(
        title=title,
        hovermode='x',
        xaxis=dict(
            title='Angle',
            ticklen=5,
            zeroline=False,
            gridwidth=2,
        ),
        yaxis=dict(
            title='Words',
            ticklen=5,
            gridwidth=2,
        ),
        showlegend=False
    )

    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename=filename)
