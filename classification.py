from sklearn import svm
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.externals import joblib
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
                              norm_pca_test_features, n_clusters=3, eps=10000,
                              method=0):
    traffic_idx = {}
    n_obs, n_features = norm_pca_features.shape
    centroids = np.array([])

    for c in range(n_clusters):
        centroids = np.append(centroids, np.mean(
            norm_pca_features[(obs_classes == c).flatten(), :], axis=0))

    centroids = centroids.reshape((n_clusters, n_features))

    cluster_method = KMeans(init=centroids, n_clusters=n_clusters) \
        if method == 0 else DBSCAN(eps=eps)
    cluster_method.fit(norm_pca_features)
    labels = cluster_method.labels_

    # Determines and quantifies the presence of each original class observation
    #  in each cluster
    clusters = np.zeros((n_clusters, n_features))
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

    # Save model
    joblib.dump(clf, 'nn_model.sav')

    result = clf.predict(norm_pca_test_features)

    for i in range(norm_pca_test_features.shape[0]):
        traffic_idx[i] = result[i]

    return traffic_idx


def accuracy_score(y_pred, y_true):
    hits = sum(y_true[i] == y_pred[i] or (y_true[i] <= 12 and y_pred[i] <= 12)
            or (y_true[i] >= 13 and y_pred[i] >= 13) 
            for i in range(len(y_true)))[0]
    return hits / len(y_true)


def main():
    unnorm_train_features, unnorm_test_features, \
    norm_pca_train_features, norm_pca_test_features, \
    traffic_classes, traffic_samples_number = profiling.profiling()

    # Save profiling data
    d = {
        'unnorm_train': unnorm_train_features,
        'unnorm_test': unnorm_test_features,
        'norm_train': norm_pca_train_features,
        'norm_test': norm_pca_test_features,
        'classes': traffic_classes,
        'samples_number': traffic_samples_number
    }

    with open('input_data.pkl', 'wb') as output:
        pickle.dump(d, output, pickle.HIGHEST_PROTOCOL)

    """
    Load data
    
    with open('input_data.pkl', 'rb') as input:
        d = pickle.load(input)

    unnorm_train_features = d['unnorm_train']
    unnorm_test_features = d['unnorm_test']
    norm_pca_train_features = d['norm_train']
    norm_pca_test_features = d['norm_test']
    traffic_classes = d['classes']
    traffic_samples_number = d['samples_number']
    """

    obs_classes = profiling.get_obs_classes(traffic_samples_number, 1,
                                            traffic_classes)

    #profiling.plot_features(unnorm_train_features, obs_classes)


    y_test = classification_gaussian_distribution(traffic_classes, obs_classes,
                                                  norm_pca_train_features,
                                                  norm_pca_test_features)

    print('GAUSSIAN acc = ', accuracy_score(list(y_test.values()), obs_classes))

    y_test = classification_svm(obs_classes, norm_pca_train_features,
                                             norm_pca_test_features, mode=1)

    print('SVM acc = ', accuracy_score(list(y_test.values()), obs_classes))

    y_test = classification_neural_networks(obs_classes, norm_pca_train_features,
                                            norm_pca_test_features)

    print('NN acc = ', accuracy_score(list(y_test.values()), obs_classes))


if __name__ == '__main__':
    main()
