import numpy as np
import pandas as pd
import spacy
import scipy.spatial.distance as dist


def get_distance_matrix(euclidean=True):
    model = spacy.load('en_vectors_glove_md')
    df = pd.read_csv('/home/tadas/words.txt')
    df = pd.DataFrame(data=list(set(list(df['word']))))

    distance_matrix = []

    for i, word in enumerate(df[0]):
        # print(i)
        vector = model(word).vector
        row = []
        for word_2 in df[0]:
            vector_2 = model(word_2).vector
            if word == word_2:
                row.append(0)
                continue
            if euclidean:
                vector = vector/np.linalg.norm(vector)
                vector_2 = vector_2/np.linalg.norm(vector_2)
                distance = np.linalg.norm(vector - vector_2)
            else:
                distance = dist.cosine(vector, vector_2)
            row.append(distance)
        distance_matrix.append(row)

    distance_matrix = np.array(distance_matrix)
    return df, distance_matrix
