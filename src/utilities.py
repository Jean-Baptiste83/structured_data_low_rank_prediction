import scipy.io as sio
import numpy as np
from scipy.sparse import lil_matrix
import igraph
import time
import pandas as pd
from tqdm import tqdm
from lenskit.matrix import sparse_ratings


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class DataHandler:
    def __init__(self, settings):
        self.n_rows = None
        self.n_cols = None
        self.tr_matrix = None
        self.tr_matrix_mask = None
        self.val_matrix = None
        self.test_matrix = None
        self.test_matrix = None
        self.tr_embedded = None
        self.tr_embedded_mask = None
        self.n_rows_embedded = None
        self.n_cols_embedded = None
        self.n_side_info = None
        self.side_info = None

        if settings.dataset == "BX-short10600":
            df = pd.read_csv("datasets/BX-short10600.csv", sep=';')
            df = df.rename(columns={'User-ID': 'user', 'ISBN': 'item', 'Book-Rating': 'rating'})
            df2 = df.copy(deep=True)
            df2['rating'] = df2['rating'] + 1

            sparse_matrix1 = sparse_ratings(df)
            sparse_matrix2 = sparse_ratings(df2)

            full_matrix = sparse_matrix1[0].to_scipy()
            full_matrix2 = sparse_matrix2[0].to_scipy()

            weights = (full_matrix2 > 0)
            full_matrix = full_matrix[:, (weights.toarray().sum(axis=0) >= 3)] # choose the books that has been anotated a least 3 times

        else:
            temp = sio.loadmat('datasets/' + settings.dataset + '.mat')
            full_matrix = temp['full_matrix'].astype(float)

        n_rows, n_cols = full_matrix.shape  # les données
        self.n_rows = n_rows
        self.n_cols = n_cols

        tr_matrix = lil_matrix((n_rows, n_cols))
        val_matrix = lil_matrix((n_rows, n_cols))
        test_matrix = lil_matrix((n_rows, n_cols))
        train_idxs = [0] * n_rows
        val_idxs = [0] * n_rows
        test_idxs = [0] * n_rows

        pbar = tqdm(range(n_rows), desc="Construction of the DataHandler")
        for c_row in pbar:
            all_ratings = np.nonzero(full_matrix[c_row:])[1]
            all_ratings_shuffled = np.random.permutation(all_ratings)
            n_all_ratings = len(all_ratings_shuffled)

            n_train = int(np.floor(
                settings.train_perc * n_all_ratings))  # pour chaque user, prendre un pourcentage pour train et l'autre pour le test
            n_val = int(np.ceil(settings.val_perc * n_all_ratings))

            train_idx = all_ratings_shuffled[:n_train]  # on récupère une partie des ratings pour le train
            val_idx = all_ratings_shuffled[n_train + 1:n_train + n_val]  # pour le test
            test_idx = all_ratings_shuffled[n_train + n_val + 1:]  # pour la validation

            train_idxs[c_row] = train_idx  # list des indices pour le train
            val_idxs[c_row] = val_idx
            test_idxs[c_row] = test_idx

            tr_matrix[c_row, train_idx] = full_matrix[c_row, train_idx]  # récupère les ratings
            val_matrix[c_row, val_idx] = full_matrix[c_row, val_idx]  # same for val
            test_matrix[c_row, test_idx] = full_matrix[c_row, test_idx]  # and test

        side_info = np.zeros((n_rows, n_cols + 1))
        side_info[:, :-1] = tr_matrix.toarray()
        side_info = side_info / np.tile(np.sqrt(1 + np.sum(tr_matrix.toarray() ** 2, 1)).reshape(n_rows, 1), (
            1, n_cols + 1))  # normalisation par sqrt((1 + sum**2)) row wise (1 to avoid dividing by 0)
        n_side_info = side_info.shape[1]

        tr_embedded, tr_embedded_mask = pairwise_embedding(tr_matrix, n_rows, n_cols)
        tr_embedded = tr_embedded
        tr_embedded_mask = tr_embedded_mask
        n_rows_embedded = n_rows
        n_cols_embedded = tr_embedded.shape[1]

        tr_matrix_mask = tr_matrix != 0
        tr_matrix_mask.astype(np.int)
        val_matrix = val_matrix
        test_matrix = test_matrix

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.tr_matrix = tr_matrix
        self.tr_matrix_mask = tr_matrix_mask
        self.val_matrix = val_matrix
        self.test_matrix = test_matrix
        self.test_matrix = test_matrix
        self.tr_embedded = tr_embedded
        self.tr_embedded_mask = tr_embedded_mask
        self.n_rows_embedded = n_rows_embedded
        self.n_cols_embedded = n_cols_embedded
        self.n_side_info = n_side_info
        self.side_info = side_info


def pairwise_embedding(y, n_rows, n_cols):  # pairwise embedding. For each y -> Ky and a mask matrix

    mask_embedded = lil_matrix(
        (n_rows, int((n_cols * (n_cols - 1) / 2))))  # matrix symtrique (il suffit de définir la diagonale supérieur
    y_embedded = lil_matrix((n_rows, int((n_cols * (n_cols - 1) / 2))))  # embeddings

    t = time.time()

    for c_row in range(n_rows):

        temp_matrix = y[c_row, :].toarray() - y[c_row, :].toarray().T
        temp_mask = np.zeros(temp_matrix.shape)
        temp_mask[np.where(temp_matrix)] = 1  # là où on n'a pas de 0 on met 1
        temp_matrix = temp_mask * temp_matrix

        iu = np.triu_indices(n_cols, 1)  # return the indices for the upper-triangle

        temp_mask_vect = temp_mask[iu]  # mask of the upper triangle
        temp_matrix_vect = temp_matrix[iu]  # values of the upper triangle

        y_embedded[c_row, temp_mask_vect > 0] = temp_matrix_vect[
            temp_mask_vect > 0]  # on copie les valeurs (yi - yj) dans y_embedded
        mask_embedded[c_row, temp_mask_vect > 0] = temp_mask_vect[temp_mask_vect > 0]  # same pour les masks

        if c_row % 100 == 0:
            print('embedding row: %5d | time: %8.3f' % (c_row, time.time() - t))

    return y_embedded, mask_embedded


def decoding(row, n_cols):
    i_upper = np.triu_indices(n_cols, 1)  # indices of the upper-triangle
    adjacency_matrix = np.zeros((n_cols, n_cols))
    adjacency_matrix[i_upper] = row  # populate the upper triangle

    a_matrix = adjacency_matrix

    a_neg = np.zeros(a_matrix.shape)  #
    a_neg[np.where(a_matrix < 0)] = a_matrix[np.where(a_matrix < 0)]  # get the negative values

    a_matrix[np.where(a_matrix < 0)] = 0  # set to 0 the negative values of the a_matrix

    i_lower = np.tril_indices(n_cols, -1)
    a_matrix[i_lower] = np.abs(a_neg.T[i_lower])  # put the absolute of negative value in the lower bound and set the
    # negative value of the upper triangle to 0

    g = igraph.Graph.Adjacency((a_matrix > 0).tolist())  # edges
    g.es['weight'] = a_matrix[a_matrix.nonzero()]  # add weights to edges

    idx_to_drop = g.feedback_arc_set(weights=g.es['weight'])

    g.delete_edges(idx_to_drop)
    adjacency_matrix = pd.DataFrame(g.get_adjacency(attribute='weight').data).values

    ordering = g.topological_sorting(mode='OUT')  # ordering of the document done with topological_sorting

    return adjacency_matrix, ordering


def pairwise_loss_separate(y, mask_embedded, data, w_matrix):
    n_rows = data.n_rows
    n_cols = data.n_cols

    def get_relative_ranking_from_ordering(ordering):
        ordering = np.array(ordering)
        ranking = np.zeros(ordering.shape)
        index_vec = np.arange(1, len(ordering) + 1)
        ranking[ordering] = index_vec
        ranking = ranking.astype(int)
        ranking = np.reshape(ranking, (len(ranking), 1))

        rel_rankings = np.sign(ranking - ranking.T)  # (sign(y_hatj - y_hati)) matrix
        return rel_rankings

    def get_rel_ratings_from_true_embedding(y_embedded, rel_rankings):
        rel_ratings = np.zeros(rel_rankings.shape)
        y_temp = y_embedded.toarray()
        iu = np.triu_indices(n_cols, 1)  # upper triangle
        rel_ratings[iu] = y_temp
        return rel_ratings  # (yi - yj)

    perf = 0

    n_rows_counter = n_rows
    t = time.time()

    for c_row in range(n_rows):  # iterate over data

        adjacency_matrix, curr_ordering = decoding(w_matrix[c_row, :] * mask_embedded[c_row, :].toarray(), n_cols)
        curr_rel_rankings = get_relative_ranking_from_ordering(curr_ordering)
        curr_rel_ratings = get_rel_ratings_from_true_embedding(y[c_row, :], curr_rel_rankings)
        temp = curr_rel_ratings * curr_rel_rankings  # (yi - yj) * (sign(y_hatj - y_hati))

        l_min = np.sum(curr_rel_ratings * -np.sign(curr_rel_ratings))  # min loss correct ordering
        l_max = np.sum(curr_rel_ratings * np.sign(curr_rel_ratings))  # max loss opposite ordering

        if (l_min == 0) and (l_max == 0):  # not take into account the data point if all the documents are
            # equivalently relevant (yi == yj for all i, j)
            n_rows_counter = n_rows_counter - 1  # substract this data points
        else:
            c_perf = (np.sum(temp) - l_min) / (l_max - l_min)  # scaled loss between 0 and 1
            perf = perf + c_perf  # add the loss

        if c_row % 100 == 0:
            print('row: %4d | time lapsed: %7.2f' % (c_row, time.time() - t))

    perf = perf / n_rows_counter  # mean loss on the entire dataset

    return perf
