from graph import DataGraph
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import math
import scipy.spatial.distance as dist

GOOD_TYPES = ['word2vec', 'glove']
GOOD_DISTANCES = ['euclidean', 'arccos']


def setup_plotly():
    plotly.tools.set_credentials_file(username='tadas.t', api_key='Ngz5K6kLcZm19NzZxH9b')


def get_word_clusters(df, dg):
    new_clusters = []
    for cluster in dg.clusters:
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


def get_dist_matrix(model, model_type, distance, file):
    if distance not in GOOD_DISTANCES:
        raise ValueError('invalid distance')

    if distance == 'arccos':
        dist_funct = lambda x, y: math.degrees(np.arccos(1 - dist.cosine(x, y)))

    if distance == 'euclidean':
        dist_funct = lambda x, y: np.linalg.norm(x-y)

    df = pd.read_csv(file)
    df = pd.DataFrame(data=list(set(list(df['word']))))

    distance_matrix = []

    for word in df[0]:
        vector = get_wordvec(word, model, model_type)
        row = []
        for word_2 in df[0]:
            if word == word_2:
                row.append(0)
                continue
            vector_2 = get_wordvec(word_2, model, model_type)
            d = dist_funct(vector, vector_2)
            row.append(d)
        distance_matrix.append(row)

    distance_matrix = np.array(distance_matrix)
    return df, distance_matrix


def produce_data(df, matrix, start, end, step):
    cluster_list = []
    for epsilon in range(start, end, step):
        dg = DataGraph(matrix, epsilon, 300)
        print('Started clustering {0}'.format(epsilon))
        dg.cluster(report_homology=True)
        clusters = get_word_clusters(df, dg)
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
            x = x_values,
            y = word_y_values[word],
            text = [str(x) for x in word_homologies[word]],
            name = word,
            hoverinfo='text+name'
        )
        data.append(trace)


    layout= go.Layout(
        title= title,
        hovermode= 'y',
        xaxis= dict(
            title= 'Angle',
            ticklen= 5,
            zeroline= False,
            gridwidth= 2,
        ),
        yaxis=dict(
            title= 'Words',
            ticklen= 5,
            gridwidth= 2,
        ),
        showlegend= False
    )

    fig= go.Figure(data=data, layout=layout)
    py.plot(fig, filename=filename)