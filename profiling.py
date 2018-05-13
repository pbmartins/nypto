import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
import warnings
import scalogram

warnings.filterwarnings('ignore')


def waitforEnter(fstop=True):
    if fstop:
        if sys.version_info[0] == 2:
            input("Press ENTER to continue.")
        else:
            input("Press ENTER to continue.")
            

def plot_traffic_class(data, name):
    plt.plot(data)
    plt.title(name)


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
    waitforEnter()


def get_obs_classes(n_obs_windows, traffic_classes):
    return np.vstack([np.ones((n_obs_windows, 1)) * t for t in traffic_classes])


def plot_features(features, traffic_classes, feature1_idx=0, feature2_idx=1):
    n_obs_windows, n_features = features.shape
    obs_classes = get_obs_classes(
        int(n_obs_windows / len(traffic_classes)), traffic_classes)
    colors = ['b', 'g', 'r']

    for i in range(n_obs_windows):
        plt.plot(features[i, feature1_idx], features[i, feature2_idx],
                 'o' + colors[int(obs_classes[i])])

    plt.show()
    waitforEnter()


def break_train_test(data, obs_window=120, slide_window=20,
                     train_percentage=0.5, random_split=True):
    n_samples, n_cols = data.shape
    n_slide_windows = int(n_samples / slide_window)
    n_obs_windows = int((n_samples - obs_window) / slide_window) + 1
    window_size = int(obs_window / slide_window)

    data_slices = data.reshape((n_slide_windows, slide_window, n_cols))
    data_obs = np.array([np.concatenate(data_slices[i:window_size+i], axis=0)
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
    features = np.array([np.hstack((
        np.mean(data[i, :, :], axis=0),
        np.median(data[i, :, :], axis=0),
        np.std(data[i, :, :], axis=0),
        stats.skew(data[i, :, :]),
        stats.kurtosis(data[i, :, :]),
        np.array(np.percentile(data[i, :, :], percentils, axis=0)).T.flatten(),
    )) for i in range(n_obs_windows)])
    
    return features


def extract_silence(data, threshold=256):
    s = [1] if data[0] <= threshold else []

    for i in range(1, len(data)):
        if data[i] <= threshold:
            if data[i-1] > threshold:
                s.append(1)
            elif data[i-1] <= threshold:
                s[-1] += 1

    return s


def extract_features_silence(data):
    features = []
    n_obs_windows, n_samples, n_cols = data.shape
    # Rebenta caso o threshold the silêncio seja 0

    for i in range(n_obs_windows):
        silence_features = np.array([])
        for c in range(n_cols):
            silence = extract_silence(data[i, :, c], threshold=250)
            silence_features = np.append(
                silence_features, [np.mean(silence), np.var(silence)])

        features.append(silence_features)

    return np.array(features)


def traffic_profiling(dataset_path, traffic_class, plot=True):
    dataset = np.loadtxt(dataset_path)

    if plot:
        plot_traffic_class(dataset, traffic_class)

    data_train, data_test = break_train_test(dataset, random_split=False)
    features = extract_features(data_train)
    test_features = extract_features(data_test)
    features_silence = extract_features_silence(data_train)
    test_features_silence = extract_features_silence(data_test)
    n_obs_windows = data_train.shape[0]

    return features, features_silence, test_features, \
           test_features_silence, n_obs_windows


def normalize_features(features, test_features):
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    normalized_test_features = scaler.fit_transform(test_features)

    pca = PCA(n_components=3, svd_solver='full')
    normalized_pca_features = pca.fit_transform(normalized_features)
    normalized_pca_test_features = pca.fit_transform(normalized_test_features)

    return normalized_pca_features, normalized_pca_test_features


def extract_traffic_features(traffic_classes, datasets_filepath):
    if len(traffic_classes) == 0 \
            or len(traffic_classes) != len(datasets_filepath):
        return None

    features = None
    features_silence = None
    test_features = None
    test_features_silence = None
    n_obs = None

    for d_idx in datasets_filepath:
        d = datasets_filepath[d_idx]
        f, fs, tf, tfs, n_obs = traffic_profiling(d, traffic_classes[d_idx])
        if features is None:
            features = np.array([f])
            features_silence = np.array([fs])
            test_features = np.array([tf])
            test_features_silence = np.array([tfs])
        else:
            features = np.vstack((features, [f]))
            features_silence = np.vstack((features_silence, [fs]))
            test_features = np.vstack((test_features, [tf]))
            test_features_silence = np.vstack((test_features_silence, [tfs]))

    print('Train Stats Features Size:', features.shape)

    plt.figure(4)
    plot_features(features, traffic_classes, 0, 2)

    print('Train Silence Features Size:', features_silence.shape)
    plt.figure(5)
    plot_features(features_silence, traffic_classes, 0, 2)

    """
    scales=range(2,256)
    plt.figure(6)

    i=0
    data=yt_train[i,:,1]
    S,scalesF=scalogram.scalogramCWT(data,scales)
    plt.plot(scalesF,S,'b')

    nObs,nSamp,nCol=browsing_train.shape
    data=browsing_train[i,:,1]
    S,scalesF=scalogram.scalogramCWT(data,scales)
    plt.plot(scalesF,S,'g')

    nObs,nSamp,nCol=mining_train.shape
    data=mining_train[i,:,1]
    S,scalesF=scalogram.scalogramCWT(data,scales)
    plt.plot(scalesF,S,'r')

    plt.show()
    waitforEnter()

    ## -- 7 -- ##
    scales=[2,4,8,16,32,64,128,256]
    features_ytW,oClass_yt=extract_featuresWavelet(yt_train,scales,Class=0)
    features_browsingW,oClass_browsing=extract_featuresWavelet(browsing_train,scales,Class=1)
    features_miningW,oClass_mining=extract_featuresWavelet(mining_train,scales,Class=2)

    featuresW=np.vstack((features_ytW,features_browsingW,features_miningW))
    t_classes=np.vstack((oClass_yt,oClass_browsing,oClass_mining))

    print('Train Wavelet Features Size:',featuresW.shape)
    plt.figure(7)
    plot_features(featuresW, traffic_classes, 3, 10)
    """

    #pca = PCA(n_components=3, svd_solver='full')
    #pca_features = pca.fit(features).transform(features)

    #plt.figure(8)
    #plot_features(pca_features, traffic_classes, 0, 2)

    # Training features
    all_features = np.hstack((features, features_silence))
    print('Train (All) Features Size:', all_features.shape)

    pca = PCA(n_components=3, svd_solver='full')
    pca.fit(all_features)
    pca_features = pca.transform(all_features)

    plt.figure(10)
    plot_features(pca_features, traffic_classes, 0, 1)

    # Testing features
    all_test_features = np.hstack((test_features, test_features_silence))
    print('Test Features Size:', all_test_features.shape)

    test_pca_features = pca.transform(all_test_features)

    return all_features, pca_features, all_test_features, test_pca_features, n_obs


def logplotFeatures(features,oClass,f1index=0,f2index=1):
    nObs,nFea=features.shape
    colors=['b','g','r']
    for i in range(nObs):
        plt.loglog(
                features[i,f1index],
                features[i,f2index],
                'o'+colors[int(oClass[i])]
                )

    plt.show()
    waitforEnter()


def extract_featuresWavelet(data,scales=[2,4,8,16,32],Class=0):
    features=[]
    nObs,nSamp,nCols=data.shape
    oClass=np.ones((nObs,1))*Class
    for i in range(nObs):
        scalo_features=np.array([])
        for c in range(nCols):
            scalo,scales=scalogram.scalogramCWT(data[i,:,c],scales)
            scalo_features=np.append(scalo_features,scalo)
            
        features.append(scalo_features)
        
    return(np.array(features),oClass)
    

def profling():
    traffic_classes = {
        0: 'YouTube',
        1: 'Browsing',
        2: 'Mining (Neoscrypt - 4T CPU)',
        #3: 'Social Networking',
        #4: 'Netflix',
    }

    datasets_filepath = {
        0: 'datasets/youtube.dat',
        1: 'datasets/browsing.dat',
        2: 'datasets/mining_4t_nicehash.dat',
        #3: 'datasets/social-network.dat',
        #4: 'datasets/netflix.dat',

    }
    plt.ion()

    all_features, pca_features, all_test_features, test_pca_features, n_obs = \
        extract_traffic_features(traffic_classes, datasets_filepath)

    return all_test_features, pca_features, all_test_features,\
           test_pca_features, n_obs


if __name__ == '__main__':
    profling()
