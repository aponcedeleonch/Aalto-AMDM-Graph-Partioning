from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture


def cluster_k_means(k_eig, k, logger):
    logger.info('Using k-means to cluster the vertices')
    kmeans = KMeans(n_clusters=k).fit(k_eig)
    logger.info('K-means finished. Returning the results')
    return kmeans.labels_


def cluster_agglomerative(k_eig, k, logger, L=None):
    logger.info('Using Agglomerative clustering to cluster the vertices')
    if L is None:
        agglomerative = AgglomerativeClustering(n_clusters=k).fit(k_eig)
    else:
        agglomerative = AgglomerativeClustering(n_clusters=k,
                                                connectivity=L).fit(k_eig)
    logger.info('Agglomerative clustering finished. Returning the results')
    return agglomerative.labels_


def cluster_gmm(k_eig, k, logger):
    logger.info('Using GMM to cluster the vertices')
    gmm = GaussianMixture(n_components=k).fit(k_eig)
    labels = gmm.predict(k_eig)
    logger.info('GMM finished. Returning the results')
    return labels
