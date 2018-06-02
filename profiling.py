import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
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


def break_train_test(data, obs_window=240, slide_window=40,
                     train_percentage=0.5, random_split=True):
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
                stats.skew(data[i, :, :]),
                stats.kurtosis(data[i, :, :]),
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


def traffic_profiling(dataset_path, traffic_class, plot=True):
    dataset = np.loadtxt(dataset_path)

    if plot:
        plot_traffic_class(dataset, traffic_class)

    scales = [2, 4, 8]
    data_train, data_test = break_train_test(dataset, random_split=True)
    empty_windows_train, features = extract_features(data_train)
    empty_windows_test, test_features = extract_features(data_test)
    features_silence = extract_features_silence(data_train, empty_windows_train)
    test_features_silence = extract_features_silence(data_test, empty_windows_test)
    features_wavelet = extract_features_wavelet(data_train, empty_windows_train, scales)
    test_features_wavelet = extract_features_wavelet(data_test, empty_windows_test, scales)
    n_obs_windows = data_train.shape[0]

    return features, features_silence, features_wavelet, test_features, \
           test_features_silence, test_features_wavelet, n_obs_windows


def normalize_features(features, test_features):
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    normalized_test_features = scaler.fit_transform(test_features)

    pca = PCA(n_components=3, svd_solver='full')
    normalized_pca_features = pca.fit(normalized_features). \
        transform(normalized_features)
    normalized_pca_test_features = pca.fit(normalized_test_features). \
        transform(normalized_test_features)

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

    # Training features
    all_features = np.hstack((features, features_silence, features_wavelet))

    # Testing features (size must be the same than the training)
    all_test_features = np.hstack((
        test_features[:features.shape[0]],
        test_features_silence[:features_silence.shape[0]],
        test_features_wavelet[:features_wavelet.shape[0]]
    ))

    # Normalize train and test features
    norm_pca_train_features, norm_pca_test_features = normalize_features(all_features,
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
        5: 'Browsing & Netflix',
        6: 'Browsing & Social Networking',
        7: 'Browsing & Youtube',
        8: 'Netflix & Social Networking',
        9: 'Netflix & YouTube',
        10: 'Social Networking & YouTube',
        11: 'VPN - Netflix',
        12: 'VPN - YouTube',
        13: 'Mining (Neoscrypt - 4T CPU)',
        14: 'Mining (Neoscrypt - 2T CPU)',
        15: 'Mining (EquiHash - 65p GPU)',
        16: 'Mining (EquiHash - 85p GPU)',
        17: 'Mining (EquiHash - 100p GPU)',
        18: 'Mining (Neoscrypt - 4T CPU) & Browsing',
        19: 'Mining (Neoscrypt - 4T CPU) & Netflix',
        20: 'Mining (Neoscrypt - 4T CPU) & Social Networking',
        21: 'Mining (Neoscrypt - 4T CPU) & Youtube',
        22: 'Mining (Neoscrypt - 2T CPU) & Browsing',
        23: 'Mining (Neoscrypt - 2T CPU) & Netflix',
        24: 'Mining (Neoscrypt - 2T CPU) & Social Networking',
        25: 'Mining (Neoscrypt - 2T CPU) & Youtube',
        26: 'Mining (Equihash - 60p GPU) & Browsing',
        27: 'Mining (Equihash - 60p GPU) & Netflix',
        28: 'Mining (Equihash - 60p GPU) & Social Networking',
        29: 'Mining (Equihash - 60p GPU) & Youtube',
        20: 'Mining (Equihash - 85p GPU) & Browsing',
        31: 'Mining (Equihash - 85p GPU) & Netflix',
        32: 'Mining (Equihash - 85p GPU) & Social Networking',
        33: 'Mining (Equihash - 85p GPU) & Youtube',
        34: 'Mining (Equihash - 100p GPU) & Browsing',
        35: 'Mining (Equihash - 100p GPU) & Netflix',
        36: 'Mining (Equihash - 100p GPU) & Social Networking',
        37: 'Mining (Equihash - 100p GPU) & Youtube',
        38: 'VPN - Mining (Neoscrypt - 4T CPU)',
        39: 'VPN - Mining (Neoscrypt - 2T CPU)',
    }

    datasets_filepath = {
        0: 'datasets/youtube.dat',
        1: 'datasets/netflix.dat',
        2: 'datasets/browsing.dat',
        3: 'datasets/social-network.dat',
        4: 'datasets/email.dat',
        5: 'vpn-datasets/browsing_netflix.dat',
        6: 'vpn-datasets/browsing_social-network.dat',
        7: 'vpn-datasets/browsing_youtube.dat',
        8: 'vpn-datasets/netflix_social-network.dat',
        9: 'vpn-datasets/netflix_youtube.dat',
        10: 'vpn-datasets/social-network_youtube.dat',
        11: 'vpn-datasets/vpn-netflix.dat',
        12: 'vpn-datasets/vpn-youtube.dat',
        13: 'datasets/mining_4t_nicehash.dat',
        14: 'datasets/mining_2t_nicehash.dat',
        15: 'datasets/mining_gpu_nicehash_equihash_1070_60p.dat',
        16: 'datasets/mining_gpu_nicehash_equihash_1080ti_85p.dat',
        17: 'datasets/mining_gpu_nicehash_equihash_1080ti_100p.dat',
        18: 'merged-datasets/mining_4t_nicehash_browsing.dat',
        19: 'merged-datasets/mining_4t_nicehash_netflix.dat',
        20: 'merged-datasets/mining_4t_nicehash_social-network.dat',
        21: 'merged-datasets/mining_4t_nicehash_youtube.dat',
        22: 'merged-datasets/mining_2t_nicehash_browsing.dat',
        23: 'merged-datasets/mining_2t_nicehash_netflix.dat',
        24: 'merged-datasets/mining_2t_nicehash_social-network.dat',
        25: 'merged-datasets/mining_2t_nicehash_youtube.dat',
        26: 'merged-datasets/mining_gpu_nicehash_equihash_1070_60p_browsing.dat',
        27: 'merged-datasets/mining_gpu_nicehash_equihash_1070_60p_netflix.dat',
        28: 'merged-datasets/mining_gpu_nicehash_equihash_1070_60p_social-network.dat',
        29: 'merged-datasets/mining_gpu_nicehash_equihash_1070_60p_youtube.dat',
        30: 'merged-datasets/mining_gpu_nicehash_equihash_1080ti_85p_browsing.dat',
        31: 'merged-datasets/mining_gpu_nicehash_equihash_1080ti_85p_netflix.dat',
        32: 'merged-datasets/mining_gpu_nicehash_equihash_1080ti_85p_social-network.dat',
        33: 'merged-datasets/mining_gpu_nicehash_equihash_1080ti_85p_youtube.dat',
        34: 'merged-datasets/mining_gpu_nicehash_equihash_1080ti_100p_browsing.dat',
        35: 'merged-datasets/mining_gpu_nicehash_equihash_1080ti_100p_netflix.dat',
        36: 'merged-datasets/mining_gpu_nicehash_equihash_1080ti_100p_social-network.dat',
        37: 'merged-datasets/mining_gpu_nicehash_equihash_1080ti_100p_youtube.dat',
        38: 'vpn-datasets/vpn-mining-4t.dat',
        39: 'vpn-datasets/vpn-mining-2t.dat',
    }
    plt.ion()

    return extract_traffic_features(traffic_classes, datasets_filepath)


if __name__ == '__main__':
    profiling()
