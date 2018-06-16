import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
import warnings
import scalogram

warnings.filterwarnings('ignore')


def wait_for_enter(fstop=True):
    if fstop:
        if sys.version_info[0] == 2:
            input("Press ENTER to continue.")
        else:
            input("Press ENTER to continue.")


def plot_traffic_class(data, name):
    plt.plot(data)
    plt.title(name)
    wait_for_enter()


def plot_3_classes(data1, name1, data2, name2, data3, name3):
    plt.subplot(3, 1, 1)
    plt.plot(data1)
    plt.title(name1)
    plt.subplot(3, 1, 2)
    plt.plot(data2)
    plt.title(name2)
    plt.subplot(3, 1, 3)
    plt.plot(data3)
    plt.title(name3)
    plt.show()
    wait_for_enter()


def get_obs_classes(traffic_samples_number, n_elems, traffic_classes):
    return np.vstack(
        [np.ones((traffic_samples_number[t], n_elems)) * t for t in traffic_classes])


def plot_features(features, obs_classes, feature1_idx=0, feature2_idx=1):
    n_obs_windows, n_features = features.shape
    colors = ['aqua', 'azure', 'black', 'brown', 'darkblue', 'darkgreen',
              'fuchsia', 'gold', 'indigo']

    for i in range(n_obs_windows):
        plt.plot(features[i, feature1_idx], features[i, feature2_idx],
                 'o', c=colors[int(obs_classes[i])])

    plt.show()


def break_train_test(data, obs_window=840, slide_window=40,
                     train_percentage=0.5, random_split=True):
    if len(data) <= obs_window:
        return np.array([data]), np.array([data])

    window_size = int(obs_window / slide_window)
    n_samples, n_cols = data.shape
    n_obs_windows = int((n_samples - obs_window) / slide_window)
    n_samples = (n_obs_windows - 1) * slide_window + obs_window
    n_slide_windows = n_obs_windows + window_size - 1
    data = data[:n_samples, :]

    data_slices = data.reshape((n_slide_windows, slide_window, n_cols))
    data_obs = np.array([np.concatenate(data_slices[i:window_size + i], axis=0)
                         for i in range(n_obs_windows)])

    order = np.random.permutation(n_obs_windows) \
        if random_split else np.arange(n_obs_windows)

    n_train_windows = int(n_obs_windows * train_percentage)
    data_train = data_obs[order[:n_train_windows], :, :]
    data_test = data_obs[order[n_train_windows:], :, :]

    return data_train, data_test


def extract_features(data):
    percentils = [75, 90, 95]
    n_obs_windows, n_samples, n_cols = data.shape
    features = []
    empty_windows = []
    for i in range(n_obs_windows):
        mean = np.mean(data[i, :, :], axis=0)
        if mean[2] == 0.0 and mean[3] == 0.0:
            empty_windows.append(i)
        else:
            features.append(np.hstack((
                mean,
                np.median(data[i, :, :], axis=0),
                np.std(data[i, :, :], axis=0),
                #stats.skew(data[i, :, :]),
                #stats.kurtosis(data[i, :, :]),
                np.array(np.percentile(data[i, :, :], percentils, axis=0)).T.flatten(),
            )))
    return empty_windows, np.array(features)


def extract_silence(data, threshold=256):
    s = [1] if data[0] <= threshold else []

    for i in range(1, len(data)):
        if data[i] <= threshold:
            if data[i - 1] > threshold:
                s.append(1)
            elif data[i - 1] <= threshold:
                s[-1] += 1

    return s[1:-1] if len(s) > 2 else [0]


def extract_features_silence(data, empty_windows):
    features = []
    n_obs_windows, n_samples, n_cols = data.shape

    for i in range(n_obs_windows):
        if i in empty_windows:
            continue
        silence_features = np.array([])
        for c in range(n_cols):
            silence = extract_silence(data[i, :, c], threshold=0)
            silence_features = np.append(
                silence_features, [np.mean(silence), np.var(silence)])

        features.append(silence_features)

    return np.array(features)


def extract_features_wavelet(data, empty_windows, scales=[2, 4, 8, 16, 32]):
    features = []
    n_obs_windows, n_samples, n_cols = data.shape

    for i in range(n_obs_windows):
        if i in empty_windows:
            continue
        scalogram_features = np.array([])
        for c in range(n_cols):
            scalo, fscales = scalogram.scalogramCWT(data[i, :, c], scales)
            scalogram_features = np.append(scalogram_features, scalo)

        features.append(scalogram_features)

    return np.array(features)


def extract_live_features(data_test):
    scales = [2, 4]
    data_train, data_test = break_train_test(
        data_test, train_percentage=0.0, random_split=False)
    empty_windows_test, test_features = extract_features(data_test)
    test_features_silence = extract_features_silence(data_test, empty_windows_test)
    test_features_wavelet = extract_features_wavelet(data_test, empty_windows_test, scales)

    return test_features, test_features_silence, test_features_wavelet


def traffic_profiling(dataset_path, traffic_class, plot=True,
                      train_percentage=0.5):
    dataset = np.loadtxt(dataset_path)

    if plot:
        plot_traffic_class(dataset, traffic_class)

    scales = [2, 4]
    data_train, data_test = break_train_test(
        dataset, train_percentage=train_percentage, random_split=False)
    empty_windows_train, features = extract_features(data_train)
    empty_windows_test, test_features = extract_features(data_test)
    features_silence = extract_features_silence(data_train, empty_windows_train)
    test_features_silence = extract_features_silence(data_test, empty_windows_test)
    features_wavelet = extract_features_wavelet(data_train, empty_windows_train, scales)
    test_features_wavelet = extract_features_wavelet(data_test, empty_windows_test, scales)
    n_obs_windows = data_train.shape[0] - len(empty_windows_test)
    
    return features, features_silence, features_wavelet, test_features, \
           test_features_silence, test_features_wavelet, n_obs_windows


def normalize_live_features(test_features):
    scaler = joblib.load('classification-model/scaler.sav')
    normalized_test_features = scaler.transform(test_features)

    pca = joblib.load('classification-model/pca_model.sav')
    normalized_pca_test_features = pca.transform(normalized_test_features)

    return normalized_pca_test_features


def normalize_train_features(features, test_features):
    scaler = StandardScaler()
    scaler.fit(features)
    normalized_features = scaler.transform(features)
    normalized_test_features = scaler.transform(test_features)

    joblib.dump(scaler, 'classification-model/scaler.sav')

    pca = PCA(n_components=25, svd_solver='full')
    pca.fit(normalized_features)
    normalized_pca_features = pca.transform(normalized_features)
    normalized_pca_test_features = pca.transform(normalized_test_features)

    joblib.dump(pca, 'classification-model/pca_model.sav')

    return normalized_pca_features, normalized_pca_test_features


def extract_traffic_features(traffic_classes, datasets_filepath):
    if len(traffic_classes) == 0 \
            or len(traffic_classes) != len(datasets_filepath):
        return None

    features = None
    features_silence = None
    features_wavelet = None
    test_features = None
    test_features_silence = None
    test_features_wavelet = None
    traffic_samples_number = None

    for d_idx in datasets_filepath:
        d = datasets_filepath[d_idx]
        f, fs, fw, tf, tfs, tfw, n_obs = \
            traffic_profiling(d, traffic_classes[d_idx], False)

        if features is None:
            features = f
            features_silence = fs
            features_wavelet = fw
            test_features = tf
            test_features_silence = tfs
            test_features_wavelet = tfw
            traffic_samples_number = [n_obs]
        else:
            features = np.concatenate((features, f))
            features_silence = np.concatenate((features_silence, fs))
            features_wavelet = np.concatenate((features_wavelet, fw))
            test_features = np.concatenate((test_features, tf))
            test_features_silence = np.concatenate((test_features_silence, tfs))
            test_features_wavelet = np.concatenate((test_features_wavelet, tfw))
            traffic_samples_number.append(n_obs)

    """
    print('Train Stats Features Size:', features.shape)
    plt.figure(4)
    plot_features(features, traffic_classes, 0, 2)

    print('Train Silence Features Size:', features_silence.shape)
    plt.figure(5)
    plot_features(features_silence, traffic_classes, 0, 2)
    """
    
    feature_size = min(features.shape[0], test_features.shape[0])
    
    # Training features
    all_features = np.hstack((
        features[:feature_size],
        features_silence[:feature_size],
        features_wavelet[:feature_size]
    ))
    
    # Testing features (size must be the same than the training)
    all_test_features = np.hstack((
        test_features[:feature_size],
        test_features_silence[:feature_size],
        test_features_wavelet[:feature_size]
    ))

    # Normalize train and test features
    norm_pca_train_features, norm_pca_test_features = normalize_train_features(all_features,
                                                                               all_test_features)

    return all_features, all_test_features, norm_pca_train_features, \
           norm_pca_test_features, traffic_classes, traffic_samples_number


def profiling():
    traffic_classes = {
        0: 'YouTube',
        1: 'Netflix',
        2: 'Browsing',
        3: 'Social Networking',
        4: 'Email',
        #5: 'Browsing & Netflix',
        #6: 'Browsing & Social Networking',
        #7: 'Browsing & Youtube',
        #8: 'Netflix & Social Networking',
        #9: 'Netflix & YouTube',
        #10: 'Social Networking & YouTube',
        5: 'VPN - Netflix',
        6: 'VPN - YouTube',
        7: 'Mining (Neoscrypt - 4T CPU)',
        8: 'Mining (Neoscrypt - 2T CPU)',
        9: 'Mining (EquiHash - 65p GPU)',
        10: 'Mining (EquiHash - 85p GPU)',
        11: 'Mining (EquiHash - 100p GPU)',
        #16: 'Mining (Neoscrypt - 4T CPU) & Browsing',
        #17: 'Mining (Neoscrypt - 4T CPU) & Netflix',
        #18: 'Mining (Neoscrypt - 4T CPU) & Social Networking',
        #19: 'Mining (Neoscrypt - 4T CPU) & Youtube',
        #20: 'Mining (Neoscrypt - 2T CPU) & Browsing',
        #21: 'Mining (Neoscrypt - 2T CPU) & Netflix',
        #22: 'Mining (Neoscrypt - 2T CPU) & Social Networking',
        #23: 'Mining (Neoscrypt - 2T CPU) & Youtube',
        #24: 'Mining (Equihash - 60p GPU) & Browsing',
        #25: 'Mining (Equihash - 60p GPU) & Netflix',
        #26: 'Mining (Equihash - 60p GPU) & Social Networking',
        #27: 'Mining (Equihash - 60p GPU) & Youtube',
        #29: 'Mining (Equihash - 85p GPU) & Browsing',
        #30: 'Mining (Equihash - 85p GPU) & Netflix',
        #31: 'Mining (Equihash - 85p GPU) & Social Networking',
        #32: 'Mining (Equihash - 85p GPU) & Youtube',
        #33: 'Mining (Equihash - 100p GPU) & Browsing',
        #34: 'Mining (Equihash - 100p GPU) & Netflix',
        #35: 'Mining (Equihash - 100p GPU) & Social Networking',
        #36: 'Mining (Equihash - 100p GPU) & Youtube',
        12: 'VPN - Mining (Neoscrypt - 4T CPU)',
        13: 'VPN - Mining (Neoscrypt - 2T CPU)',
    }

    datasets_filepath = {
        0: 'datasets/youtube.dat',
        1: 'datasets/netflix.dat',
        2: 'datasets/browsing.dat',
        3: 'datasets/social-network.dat',
        4: 'datasets/email.dat',
        #5: 'merged-datasets/browsing_netflix.dat',
        #6: 'merged-datasets/browsing_social-network.dat',
        #7: 'merged-datasets/browsing_youtube.dat',
        #8: 'merged-datasets/netflix_social-network.dat',
        #9: 'merged-datasets/netflix_youtube.dat',
        #10: 'merged-datasets/social-network_youtube.dat',
        5: 'vpn-datasets/vpn-netflix.dat',
        6: 'vpn-datasets/vpn-youtube.dat',
        7: 'datasets/mining_4t_nicehash.dat',
        8: 'datasets/mining_2t_nicehash.dat',
        9: 'datasets/mining_gpu_nicehash_equihash_1070_60p.dat',
        10: 'datasets/mining_gpu_nicehash_equihash_1080ti_85p.dat',
        11: 'datasets/mining_gpu_nicehash_equihash_1080ti_100p.dat',
        #16: 'merged-datasets/mining_4t_nicehash_browsing.dat',
        #17: 'merged-datasets/mining_4t_nicehash_netflix.dat',
        #18: 'merged-datasets/mining_4t_nicehash_social-network.dat',
        #19: 'merged-datasets/mining_4t_nicehash_youtube.dat',
        #20: 'merged-datasets/mining_2t_nicehash_browsing.dat',
        #21: 'merged-datasets/mining_2t_nicehash_netflix.dat',
        #22: 'merged-datasets/mining_2t_nicehash_social-network.dat',
        #23: 'merged-datasets/mining_2t_nicehash_youtube.dat',
        #24: 'merged-datasets/mining_gpu_nicehash_equihash_1070_60p_browsing.dat',
        #25: 'merged-datasets/mining_gpu_nicehash_equihash_1070_60p_netflix.dat',
        #26: 'merged-datasets/mining_gpu_nicehash_equihash_1070_60p_social-network.dat',
        #27: 'merged-datasets/mining_gpu_nicehash_equihash_1070_60p_youtube.dat',
        #29: 'merged-datasets/mining_gpu_nicehash_equihash_1080ti_85p_browsing.dat',
        #30: 'merged-datasets/mining_gpu_nicehash_equihash_1080ti_85p_netflix.dat',
        #31: 'merged-datasets/mining_gpu_nicehash_equihash_1080ti_85p_social-network.dat',
        #32: 'merged-datasets/mining_gpu_nicehash_equihash_1080ti_85p_youtube.dat',
        #33: 'merged-datasets/mining_gpu_nicehash_equihash_1080ti_100p_browsing.dat',
        #34: 'merged-datasets/mining_gpu_nicehash_equihash_1080ti_100p_netflix.dat',
        #35: 'merged-datasets/mining_gpu_nicehash_equihash_1080ti_100p_social-network.dat',
        #36: 'merged-datasets/mining_gpu_nicehash_equihash_1080ti_100p_youtube.dat',
        12: 'vpn-datasets/vpn-mining-4t.dat',
        13: 'vpn-datasets/vpn-mining-2t.dat',
    }
    plt.ion()

    return extract_traffic_features(traffic_classes, datasets_filepath)


if __name__ == '__main__':
    profiling()
