from sklearn.cluster import DBSCAN
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import numpy as np
import profiling


def get_centroids(traffic_classes, features):
    n_obs = int(features.shape[0] / len(traffic_classes))
    obs_classes = profiling.get_obs_classes(n_obs, 1, traffic_classes)
    centroids = {t: np.mean(features[(obs_classes == t).flatten(), :], axis=0)
                 for t in traffic_classes}
    print('All Features Centroids:\n', centroids)
    return centroids


def get_covariances(traffic_classes, features):
    n_obs = int(features.shape[0] / len(traffic_classes))
    obs_classes = profiling.get_obs_classes(n_obs, 1, traffic_classes)
    centroids = {t: np.cov(features[(obs_classes == t).flatten(), :], rowvar=0)
                 for t in traffic_classes}
    print('All Features Covariances:\n', centroids)
    return centroids


def distance(centroid, point):
    return np.sqrt(np.sum(np.square(point - centroid)))


def classification_distances(traffic_classes, centroids, test_features):
    print('\n-- Classification based on Distances --')
    n_obs, n_features = test_features.shape
    traffic_idx = {}

    for i in range(n_obs):
        w = test_features[i]
        distances = [distance(w, centroids[c]) for c in centroids]
        distances_perc = distances / np.sum(distances)
        t_idx = np.argsort(distances)[0]
        traffic_idx[i] = t_idx

        print('Obs: {:2}: Normalized Distances to Centroids: '
              '[{:.4f},{:.4f},{:.4f}] -> Classification: {} -> {}'.format(
                i, *distances_perc, t_idx, traffic_classes[t_idx]))

    return traffic_idx


def classification_gaussian_distribution(traffic_classes, pca_features,
                                         test_pca_features):
    print('\n-- Classification based on Multivariate PDF (PCA Features) --')
    n_obs, n_features = test_pca_features.shape
    means = get_centroids(traffic_classes, pca_features)
    covs = get_covariances(traffic_classes, pca_features)
    traffic_idx = {}

    for i in range(n_obs):
        w = test_pca_features[i, :]
        probs = np.array([multivariate_normal.pdf(w, means[t], covs[t])
                          for t in traffic_classes])
        t_idx = np.argsort(probs)[-1]
        traffic_idx[i] = t_idx

        #print('Obs: {:2}: Probabilities: [{:.4e},{:.4e}] -> '
        #      'Classification: {} -> {}'.format(i, *probs, t_idx,
        #                                        traffic_classes[t_idx]))

    return traffic_idx


def classification_clustering(traffic_classes, norm_pca_features,
                              norm_pca_test_features, n_clusters=3, eps=10000,
                              method=0):
    print('\n-- Classification based on Clustering (Kmeans) --')
    traffic_idx = {}
    n_obs, n_features = norm_pca_features.shape
    n_obs = int(n_obs / len(traffic_classes))
    obs_classes = profiling.get_obs_classes(n_obs, 1, traffic_classes)
    centroids = np.array([])

    for c in range(n_clusters):
        centroids = np.append(centroids, np.mean(
            norm_pca_features[(obs_classes == c).flatten(), :], axis=0))

    centroids = centroids.reshape((len(traffic_classes), n_features))
    print('PCA (pca_features) Centroids:\n', centroids)

    cluster_method = KMeans(init=centroids, n_clusters=n_clusters) \
        if method == 0 else DBSCAN(eps=eps)
    cluster_method.fit(norm_pca_features)
    labels = cluster_method.labels_

    # Determines and quantifies the presence of each original class observation in each cluster
    clusters = np.zeros((len(traffic_classes), n_features))
    for cluster in range(n_clusters):
        aux = obs_classes[(labels == cluster)]
        for c in range(n_clusters):
            clusters[cluster, c] = np.sum(aux == c)

    cluster_probs = clusters / np.sum(clusters, axis=1)[:, np.newaxis]

    for i in range(norm_pca_test_features.shape[0]):
        x = norm_pca_test_features[i, :].reshape((1, n_features))
        label = cluster_method.predict(x)
        t_probs = 100 * cluster_probs[label, :].flatten()
        t_idx = np.argsort(t_probs)[-1]
        traffic_idx[i] = t_idx

        #print('Obs: {:2}: Probabilities beeing in each class: '
        #      '[{:.2f}%,{:.2f}%] -> Classification: {} -> {}'.format(
        #    i, *t_probs, t_idx, traffic_classes[t_idx]))

    return traffic_idx


def classification_svm(traffic_classes, norm_features, norm_test_features, mode=0):
    print('\n-- Classification based on Support Vector Machines --')
    traffic_idx = {}
    modes = {
        0: {'name': 'SVC', 'func': svm.SVC(kernel='linear')},
        1: {'name': 'Kernel RBF', 'func': svm.SVC(kernel='rbf')},
        2: {'name': 'Kernel Poly', 'func': svm.SVC(kernel='poly', degree=2)},
        3: {'name': 'Linear SVC', 'func': svm.LinearSVC()}
    }
    n_obs, n_features = norm_features.shape
    obs_classes = profiling.get_obs_classes(n_obs, n_features, traffic_classes)

    modes[mode]['func'].fit(norm_features, obs_classes)
    result = modes[mode]['func'].predict(norm_features)
    print('class (from test PCA features with {}):'.format(modes[mode]['name']),
          result)

    for i in range(norm_test_features.shape[0]):
        traffic_idx[i] = result[i]
        print('Obs: {:2}: {} -> {}'.format(i, modes[mode],
                                           traffic_classes[result[i]]))

    return traffic_idx


def classification_neural_networks(traffic_classes, norm_pca_features,
                                   norm_pca_test_features, alpha=1,
                                   max_iter=100000, hidden_layer_size=100):
    print('\n-- Classification based on Neural Networks --')
    traffic_idx = {}
    n_obs, n_features = norm_pca_features.shape
    obs_classes = profiling.get_obs_classes(n_obs, n_features, traffic_classes)
    clf = MLPClassifier(
        solver='lbfgs',
        alpha=alpha,
        hidden_layer_sizes=(hidden_layer_size,),
        max_iter=max_iter
    )
    clf.fit(norm_pca_features, obs_classes)
    result = clf.predict(norm_pca_features)
    print('class (from test PCA):', result)

    for i in range(norm_pca_test_features.shape[0]):
        traffic_idx[i] = result[i]
        print('Obs: {:2}: Classification->{}'.format(i, traffic_classes[result[i]]))

    return traffic_idx


def main():
    traffic_classes, norm_pca_features, \
    norm_pca_test_features, n_obs = profiling.profiling()

    """
    x = classification_gaussian_distribution(traffic_classes,
                                             norm_pca_features,
                                             norm_pca_test_features)
    print(x)
    """
    x = classification_clustering(traffic_classes, norm_pca_features,
                                  norm_pca_test_features, n_clusters=2)
    print(x)

if __name__ == '__main__':
    main()