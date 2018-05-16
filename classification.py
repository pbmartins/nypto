from sklearn import metrics
from sklearn import svm
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from scipy.stats import multivariate_normal
import numpy as np
import pickle
import profiling


def get_centroids(traffic_classes, obs_classes, features):
    centroids = {t: np.mean(features[(obs_classes == t).flatten(), :], axis=0)
                 for t in traffic_classes}
    return centroids


def get_covariances(traffic_classes, obs_classes, features):
    centroids = {t: np.cov(features[(obs_classes == t).flatten(), :], rowvar=0)
                 for t in traffic_classes}
    return centroids


def distance(centroid, point):
    return np.sqrt(np.sum(np.square(point - centroid)))


def classification_distances(centroids, test_features):
    n_obs, n_features = test_features.shape
    traffic_idx = {}

    for i in range(n_obs):
        w = test_features[i]
        distances = [distance(w, centroids[c]) for c in centroids]
        t_idx = np.argsort(distances)[0]
        traffic_idx[i] = t_idx

    return traffic_idx


def classification_gaussian_distribution(traffic_classes, obs_classes, pca_features,
                                         test_pca_features):
    n_obs, n_features = test_pca_features.shape
    means = get_centroids(traffic_classes, obs_classes, pca_features)
    covs = get_covariances(traffic_classes, obs_classes, pca_features)
    traffic_idx = {}

    for i in range(n_obs):
        w = test_pca_features[i, :]
        probs = np.array([multivariate_normal.pdf(w, means[t], covs[t])
                          for t in traffic_classes])
        t_idx = np.argsort(probs)[-1]
        traffic_idx[i] = t_idx

    return traffic_idx


def classification_clustering(traffic_classes, obs_classes, norm_pca_features,
                              norm_pca_test_features, n_clusters=9, eps=10000,
                              method=0):
    traffic_idx = {}
    n_obs, n_features = norm_pca_features.shape
    centroids = np.array([])

    for c in range(n_clusters):
        centroids = np.append(centroids, np.mean(
            norm_pca_features[(obs_classes == c).flatten(), :], axis=0))

    centroids = centroids.reshape((len(traffic_classes), n_features))

    cluster_method = KMeans(init=centroids, n_clusters=n_clusters) \
        if method == 0 else DBSCAN(eps=eps)
    cluster_method.fit(norm_pca_features)
    labels = cluster_method.labels_

    # Determines and quantifies the presence of each original class observation
    #  in each cluster
    clusters = np.zeros((len(traffic_classes), n_features))
    for cluster in range(n_clusters):
        aux = obs_classes[(labels == cluster)]
        for c in range(n_features):
            clusters[cluster, c] = np.sum(aux == c)

    cluster_probs = clusters / np.sum(clusters, axis=1)[:, np.newaxis]

    for i in range(norm_pca_test_features.shape[0]):
        x = norm_pca_test_features[i, :].reshape((1, n_features))
        label = cluster_method.predict(x)
        t_probs = 100 * cluster_probs[label, :].flatten()
        t_idx = np.argsort(t_probs)[-1]
        traffic_idx[i] = t_idx

    return traffic_idx


def classification_svm(obs_classes, norm_features,
                       norm_test_features, mode=0):
    traffic_idx = {}
    modes = {
        0: {'name': 'SVC', 'func': svm.SVC(kernel='linear')},
        1: {'name': 'Kernel RBF', 'func': svm.SVC(kernel='rbf')},
        2: {'name': 'Kernel Poly', 'func': svm.SVC(kernel='poly', degree=2)},
        3: {'name': 'Linear SVC', 'func': svm.LinearSVC()}
    }

    modes[mode]['func'].fit(norm_features, obs_classes)
    result = modes[mode]['func'].predict(norm_test_features)

    for i in range(norm_test_features.shape[0]):
        traffic_idx[i] = result[i]

    return traffic_idx


def classification_neural_networks(obs_classes, norm_pca_features,
                                   norm_pca_test_features, alpha=0.1,
                                   max_iter=100000, hidden_layer_size=1000):

    traffic_idx = {}
    clf = MLPClassifier(
        solver='lbfgs',
        alpha=alpha,
        hidden_layer_sizes=(hidden_layer_size,),
        max_iter=max_iter
    )
    clf.fit(norm_pca_features, obs_classes)
    result = clf.predict(norm_pca_test_features)

    for i in range(norm_pca_test_features.shape[0]):
        traffic_idx[i] = result[i]

    return traffic_idx


def main():
    """
    traffic_classes, norm_pca_features, norm_pca_test_features, \
    traffic_samples_number = profiling.profiling()

    d = {
        'classes': traffic_classes,
        'train': norm_pca_features,
        'test': norm_pca_test_features,
        'number': traffic_samples_number
    }
    with open('input_data.pkl', 'wb') as output:
        pickle.dump(d, output, pickle.HIGHEST_PROTOCOL)
    """

    with open('input_data.pkl', 'rb') as input:
        d = pickle.load(input)

    traffic_classes = d['classes']
    norm_pca_features = d['train']
    norm_pca_test_features = d['test']
    traffic_samples_number = d['number']

    # TODO: traffic_classes with all categories instead of joined mining datasets

    obs_classes = profiling.get_obs_classes(traffic_samples_number, 1,
                                            traffic_classes)

    """
    y_test = classification_gaussian_distribution(traffic_classes, obs_classes,
                                                  norm_pca_features,
                                                  norm_pca_test_features)

    print('GAUSSIAN acc = ',
          metrics.accuracy_score(list(y_test.values()), obs_classes))

    y_test = classification_clustering(traffic_classes, obs_classes, norm_pca_features,
                                norm_pca_test_features)

    print('KMeans acc = ',
          metrics.accuracy_score(list(y_test.values()), obs_classes))

    """

    y_test = classification_svm(obs_classes, norm_pca_features,
                                             norm_pca_test_features, mode=1)

    print('SVM acc = ',
          metrics.accuracy_score(list(y_test.values()), obs_classes))

    y_test = classification_neural_networks(obs_classes, norm_pca_features,
                                            norm_pca_test_features)

    print('NN acc = ',
          metrics.accuracy_score(list(y_test.values()), obs_classes))


if __name__ == '__main__':
    main()