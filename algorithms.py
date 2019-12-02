from scipy import sparse
from sklearn.preprocessing import normalize
import numpy as np
import networkx as nx
import pickle
from resources import COMP_FOLDER, score_function, correct_cluster_labels
from clusterings import cluster_k_means, cluster_agglomerative, cluster_gmm, cluster_k_means_modified


def get_stored_L_eigv(G_meta, norm, k, logger):
    eigv_file = '%s_%s_%d_eigv.pkl' % (G_meta['name'], norm, k)
    lap_file = '%s_%s_%d_lap.pkl' % (G_meta['name'], norm, k)

    eigv_path = COMP_FOLDER + '/' + eigv_file
    lap_path = COMP_FOLDER + '/' + lap_file

    try:
        logger.debug('Looking for eigenvector file: %s' % (eigv_path))
        logger.debug('Looking for laplacian file: %s' % (lap_path))

        with open(eigv_path, 'rb') as file:
            eigenvec = pickle.load(file=file)
            logger.info('Eigenvector file found')

        with open(lap_path, 'rb') as file:
            L = pickle.load(file=file)
            logger.info('Laplacian file found')

    except FileNotFoundError:
        eigenvec = None
        L = None
        logger.info('Eigenvector or Laplacian file not found')

    return L, eigenvec


def store_L_eigv(G_meta, norm, k, eigv, L, logger):
    eigv_file = '%s_%s_%d_eigv.pkl' % (G_meta['name'], norm, k)
    lap_file = '%s_%s_%d_lap.pkl' % (G_meta['name'], norm, k)

    eigv_path = COMP_FOLDER + '/' + eigv_file
    lap_path = COMP_FOLDER + '/' + lap_file

    with open(eigv_path, 'wb') as file:
        pickle.dump(eigv, file=file, protocol=4)
        logger.info('Eigenvector information dumped to file')
        logger.debug('Writren eigevector file: %s' % (eigv_path))

    with open(lap_path, 'wb') as file:
        pickle.dump(L, file=file, protocol=4)
        logger.info('Laplacian information dumped to file')
        logger.debug('Writren Laplacian file: %s' % (lap_path))


def laplacian_and_k_eigenval_eigenvec(G, k, norm_decide, logger):
    # Get the Laplacian matrix from the graph
    if('norm' in norm_decide):
        logger.info('Getting Normalized Laplacian matrix')
        L = nx.normalized_laplacian_matrix(G)
    else:
        logger.info('Getting Laplacian matrix')
        L = nx.laplacian_matrix(G)
    L_double = L.asfptype()
    # Get the eigenvalues and eigenvectors
    logger.info('Getting eigenvalues and eigenvectors of Laplacian')
    # Note use of function eigsh over eig.
    # eigsh for real symmetric matrix and only k values
    eigenval, eigenvec = sparse.linalg.eigsh(L_double, which='SM', k=k, ncv=5*k)
    if (norm_decide == 'norm_eig'):
        logger.info('Normalizing eigenvec matrix')
        eigenvec = normalize(eigenvec, axis=1, norm='l2')
    logger.info('Finished. Returning eigenvalues, eigenvectors and Laplacian')
    return L, eigenval, eigenvec


def unorm(G, G_meta, clustering, n, logger):
    k = G_meta['k']
    # Get Laplacian, k eigenvalues and eigenvectors of it
    L, k_eigenvec = get_stored_L_eigv(G_meta, 'unorm', k, logger)
    if k_eigenvec is None:
        logger.info('Going to compute Laplacian and eigenvectors')
        L, _, k_eigenvec = laplacian_and_k_eigenval_eigenvec(G, k+1, 'u', logger)
        store_L_eigv(G_meta, 'unorm', k, k_eigenvec, L, logger)

    k_eigenvec = k_eigenvec[:, 1:]
    logger.debug("Shape of K eigenvector matrix: %s" % (k_eigenvec.shape, ))
    logger.debug('K-Eigenvectors')
    logger.debug(k_eigenvec)
    if (clustering == 'Kmeans'):
        # Cluster using k-means
        _, cluster_labels = cluster_k_means(k_eigenvec, k, logger)
    if (clustering =='Kmeans_modified'):
        #Cluster usign modified Kmeans
        logger.info('To merge %d nodes per cluster' % (n))
        cluster_labels = cluster_k_means_modified(G, n, k_eigenvec, k, logger)
        return cluster_labels
    if (clustering == "Gmm"):
        # Cluster using gmm
        cluster_labels = cluster_gmm(k_eigenvec, k, logger)
    if (clustering == "Agglomerative"):
        # Cluster using agglomerative algorithm
        cluster_labels = cluster_agglomerative(k_eigenvec, k, logger, L)

    cluster_labels = correct_cluster_labels(G, cluster_labels)
    return cluster_labels


def norm_lap(G, G_meta, clustering, n, logger):
    k = G_meta['k']
    # Get Laplacian, k eigenvalues and eigenvectors of it
    L, k_eigenvec = get_stored_L_eigv(G_meta, 'norm', k, logger)
    if k_eigenvec is None:
        logger.info('Going to compute Laplacian and eigenvectors')
        L, _, k_eigenvec = laplacian_and_k_eigenval_eigenvec(G, k+1, 'norm', logger)
        store_L_eigv(G_meta, 'norm', k, k_eigenvec, L, logger)
    k_eigenvec = k_eigenvec[:, 1:]
    logger.debug("Shape of K eigenvector matrix: %s" % (k_eigenvec.shape, ))
    logger.debug('K-Eigenvectors')
    logger.debug(k_eigenvec)
    if (clustering == 'Kmeans'):
        # Cluster using k-means
        _, cluster_labels = cluster_k_means(k_eigenvec, k, logger)
    if (clustering =='Kmeans_modified'):
        #Cluster usign modified Kmeans
        logger.info('To merge %d nodes per cluster' % (n))
        cluster_labels = cluster_k_means_modified(G, n, k_eigenvec, k, logger)
        return cluster_labels
    if (clustering == "Gmm"):
        # Cluster using gmm
        cluster_labels = cluster_gmm(k_eigenvec, k, logger)
    if (clustering == "Agglomerative"):
        # Cluster using agglomerative algorithm
        cluster_labels = cluster_agglomerative(k_eigenvec, k, logger, L)

    cluster_labels = correct_cluster_labels(G, cluster_labels)

    return cluster_labels


def norm_eig(G, G_meta, clustering, n, logger):
    k = G_meta['k']
    # Get Laplacian, k eigenvalues and eigenvectors of it
    L, k_eigenvec = get_stored_L_eigv(G_meta, 'norm', k, logger)
    if k_eigenvec is None:
        logger.info('Going to compute Laplacian and eigenvectors')
        L, _, k_eigenvec = laplacian_and_k_eigenval_eigenvec(G_meta, k+1, 'norm_eig', logger)
        store_L_eigv(G_meta, 'norm', k, k_eigenvec, L, logger)
    k_eigenvec = k_eigenvec[:, 1:]
    logger.debug("Shape of K eigenvector matrix: %s" % (k_eigenvec.shape, ))
    logger.debug('K-Eigenvectors')
    logger.debug(k_eigenvec)
    if (clustering == 'Kmeans'):
        # Cluster using k-means
        _, cluster_labels = cluster_k_means(k_eigenvec, k, logger)
    if (clustering =='Kmeans_modified'):
        #Cluster usign modified Kmeans
        logger.info('To merge %d nodes per cluster' % (n))
        cluster_labels = cluster_k_means_modified(G, n, k_eigenvec, k, logger)
        return cluster_labels
    if (clustering == "Gmm"):
        # Cluster using gmm
        cluster_labels = cluster_gmm(k_eigenvec, k, logger)
    if (clustering == "Agglomerative"):
        # Cluster using agglomerative algorithm
        cluster_labels = cluster_agglomerative(k_eigenvec, k, logger, L)

    cluster_labels = correct_cluster_labels(G, cluster_labels)

    return cluster_labels


def recursive(G, k, c, clustering, n, logger):
    if (k >= 2):
        # Get Laplacian, 2 eigenvalues and eigenvectors
        L, _, k_eigenvec = laplacian_and_k_eigenval_eigenvec(G, 2, 'norm', logger)
        logger.debug("Shape of K eigenvector matrix: %s" % (k_eigenvec.shape, ))
        # Cluster using k-means and the second smallest eigenvector
        eigenvec_2 = k_eigenvec[:, 1].reshape(-1, 1)
        # eigenvec_2 = k_eigenvec
        if (clustering == 'Kmeans'):
            # Cluster using k-means
            _, cluster_labels = cluster_k_means(eigenvec_2, 2, logger)
        if (clustering =='Kmeans_modified'):
            #Cluster usign modified Kmeans
            logger.info('To merge %d nodes per cluster' % (n))
            cluster_labels = cluster_k_means_modified(G, n, k_eigenvec, k, logger)
            return cluster_labels
        if (clustering == "Gmm"):
            # Cluster using gmm
            cluster_labels = cluster_gmm(eigenvec_2, 2, logger)
        if (clustering == "Agglomerative"):
            # Cluster using agglomerative algorithm
            cluster_labels = cluster_agglomerative(k_eigenvec, k, logger, L)
        # Nodes of the biggest cluster
        logger.debug("Graph partition")
        all_nodes = list(G)
        n_all = len(all_nodes)
        b_cluster = sum(cluster_labels)
        logger.info('Remaining iterations: %d.' % (k-1))
        if (b_cluster > n_all/2):
            indicator = 1
            indicator2 = 0
        else:
            indicator = 0
            indicator2 = 1
        nodes = [all_nodes[i] for i, label in enumerate(cluster_labels) if label==indicator]
        accepted_cluster = [all_nodes[i] for i, label in enumerate(cluster_labels) if label==indicator2]
        subgraph = G.subgraph(nodes)
        c[k-1] = accepted_cluster
        return recursive(subgraph, k-1, c, clustering, n, logger)
        # return c
    c[k-1] = list(G)
    return c


def hagen_kahng(G, G_meta, logger):
    k = G_meta['k']
    # Throws an execption if k!=2
    if k != 2:
        raise ValueError(('Hagen Kahng algorithm only works with k=2.'
                          'Trying to execute k=%d') % (k))
    # Get the Unormalized Laplacian matrix
    _, k_eigenvec = get_stored_L_eigv(G_meta, 'unorm', k, logger)
    if k_eigenvec is None:
        logger.info('Going to compute Laplacian and eigenvectors')
        L, _, k_eigenvec = laplacian_and_k_eigenval_eigenvec(G, k+1, 'u', logger)
        store_L_eigv(G_meta, 'unorm', k, k_eigenvec, L, logger)

    logger.debug("Shape of K eigenvector matrix: %s" % (k_eigenvec.shape, ))
    logger.debug('K-Eigenvectors')
    logger.debug(k_eigenvec)
    # Getting only the second eigenvector
    logger.info("Getting only second eigenvector")
    eigv_2 = k_eigenvec[:, 1]
    logger.debug(eigv_2)

    # Executing the Hagen Kahng algorithm
    cluster_labels = hagen_kahng_ratio_cut(eigv_2, G, logger)

    return cluster_labels


def hagen_kahng_ratio_cut(eigv_2, G, logger):
    logger.info('Executing Hagen Kahng algorithm')
    logger.debug('Sorting second eigenvector')
    # Getting the indexes for sorting the second eigenvector
    ordered_eigv = np.argsort(eigv_2)
    logger.debug(eigv_2[ordered_eigv])
    # Initialize an array to keep the results for every cut
    r_results = np.zeros(G.number_of_nodes()-1)
    # Make all possible cuts over the second eigenvector
    for i in range(len(r_results)):
        # Putting i nodes in the first cluster
        cluster_1_num = i+1
        # Putting V-i nodes in the second cluster
        cluster_2_num = G.number_of_nodes()-cluster_1_num
        logger.debug('Iteration of Hagen Kahng: %d' % (i))
        logger.debug('Nodes in Cluster 1: %d' % (cluster_1_num))
        logger.debug('Nodes in Cluster 2: %d' % (cluster_2_num))
        cluster_1 = np.zeros(cluster_1_num)
        cluster_2 = np.ones(cluster_2_num)
        clustered = np.append(cluster_1, cluster_2)
        # Ordering the cut according to the ordered eigenvector
        clustered = clustered[ordered_eigv]
        # Getting the score for that cut
        r_results[i] = score_function(clustered, k=2, G=G, logger=logger)
        logger.debug('Scoring function results: %.10f' % (r_results[i]))
    # Get the best score from all the cuts
    ideal_cut = np.argmin(r_results) + 1
    # Constructing again the way the cut was made
    clustered = np.append(np.zeros(ideal_cut), np.ones(G.number_of_nodes()-ideal_cut))
    clustered = clustered[ordered_eigv]
    logger.info('Best scoring function found Hagen Kahng: %.10f' % (r_results[ideal_cut-1]))
    logger.info('Cluster 1 size: %d' % (ideal_cut))
    logger.info('Cluster 2 size: %d' % (G.number_of_nodes()-ideal_cut))
    return clustered
