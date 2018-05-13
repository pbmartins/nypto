import numpy as np
import scipy.stats as stats
import scipy.signal as signal
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import time
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

    for i in range(n_obs_windows):
        silence_features = np.array([])
        for c in range(n_cols):
            silence = extract_silence(data[i, :, c], threshold=250)
            silence_features = np.append(
                silence_features, [np.mean(silence), np.var(silence)])

        features.append(silence_features)

    return np.array(features)



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
    
## -- 11 -- ##
def distance(c,p):
    return(np.sqrt(np.sum(np.square(p-c))))

########### Main Code #############
def main():
    traffic_classes = {
        0: 'YouTube',
        1: 'Browsing',
        2: 'Mining',
        #3: 'Social Networking',
        #4: 'Netflix',
    }
    plt.ion()

    # -- 1 --
    youtube = np.loadtxt('datasets/youtube.dat')
    browsing = np.loadtxt('datasets/browsing.dat')
    mining = np.loadtxt('datasets/mining_4t_nicehash.dat')

    plt.figure(1)
    plot_3_classes(youtube, 'YouTube', browsing, 'Browsing', mining, 'Mining')

    # -- 2 --
    yt_train, yt_test = break_train_test(youtube, random_split=False)
    browsing_train, browsing_test = break_train_test(browsing, random_split=False)
    mining_train, mining_test = break_train_test(mining, random_split=False)

    """
    plt.figure(2)
    plt.subplot(3, 1, 1)
    for i in range(10):
        plt.plot(yt_train[i, :, 0], 'b')
        plt.plot(yt_train[i, :, 1], 'g')
    plt.title('YouTube')
    plt.ylabel('Bytes/sec')
    plt.subplot(3, 1, 2)
    for i in range(10):
        plt.plot(browsing_train[i, :, 0], 'b')
        plt.plot(browsing_train[i, :, 1], 'g')
    plt.title('Browsing')
    plt.ylabel('Bytes/sec')
    plt.subplot(3, 1, 3)
    for i in range(10):
        plt.plot(mining_train[i, :, 0], 'b')
        plt.plot(mining_train[i, :, 1], 'g')
    plt.title('Mining')
    plt.ylabel('Bytes/sec')
    plt.show()
    waitforEnter()
    """

    features_yt = extract_features(yt_train)
    features_browsing = extract_features(browsing_train)
    features_mining = extract_features(mining_train)

    features = np.vstack((features_yt, features_browsing, features_mining))

    print('Train Stats Features Size:', features.shape)

    plt.figure(4)
    plot_features(features, traffic_classes, 0, 2)

    features_silence_yt = extract_features_silence(yt_train)
    features_silence_browsing = extract_features_silence(browsing_train)
    features_silence_mining = extract_features_silence(mining_train)

    features_silence = np.vstack((features_silence_yt,
                                  features_silence_browsing,
                                  features_silence_mining))

    print('Train Silence Features Size:', features_silence.shape)
    plt.figure(5)
    plot_features(features_silence, traffic_classes, 0, 2)

    """
    ## -- 6 -- ##
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

    ## -- 9 -- ##
    """
    pca = PCA(n_components=3, svd_solver='full')
    pca_features = pca.fit(featuresW).transform(featuresW)

    plt.figure(9)
    plot_features(pca_features, traffic_classes, 0, 2)
    """

    all_features = np.hstack((features, features_silence))
    print('Train (All) Features Size:', all_features.shape)

    # Rebenta caso o threshold the silêncio seja 0
    #print(np.where(np.isnan(features_silence)))
    #print(features_silence[149].shape)
    #print(features_silence[149])
    #print(np.all(np.isfinite(all_features)))

    pca = PCA(n_components=3, svd_solver='full')
    pca_features = pca.fit(all_features).transform(all_features)

    plt.figure(10)
    plot_features(pca_features, traffic_classes, 0, 1)

    centroids = {}
    obs_classes = get_obs_classes(yt_train.shape[0], traffic_classes)
    for c in range(len(traffic_classes)):
        p_class = (obs_classes == c).flatten()
        centroids.update({c: np.mean(all_features[p_class, :], axis=0)})

    print('All Features Centroids:\n', centroids)
    waitforEnter()

    testFeatures_yt,oClass_yt=extract_features(yt_test,Class=0)
    testFeatures_browsing,oClass_browsing=extract_features(browsing_test,Class=1)
    testFeatures_mining,oClass_mining=extract_features(mining_test,Class=2)
    testFeatures=np.vstack((testFeatures_yt,testFeatures_browsing,testFeatures_mining))

    testFeatures_ytS,oClass_yt=extract_features_silence(yt_test, Class=0)
    testFeatures_browsingS,oClass_browsing=extract_features_silence(browsing_test, Class=1)
    testFeatures_miningS,oClass_mining=extract_features_silence(mining_test, Class=2)
    testFeaturesS=np.vstack((testFeatures_ytS,testFeatures_browsingS,testFeatures_miningS))

    testFeatures_ytW,oClass_yt=extract_featuresWavelet(yt_test,scales,Class=0)
    testFeatures_browsingW,oClass_browsing=extract_featuresWavelet(browsing_test,scales,Class=1)
    testFeatures_miningW,oClass_mining=extract_featuresWavelet(mining_test,scales,Class=2)
    testFeaturesW=np.vstack((testFeatures_ytW,testFeatures_browsingW,testFeatures_miningW))

    alltestFeatures=np.hstack((testFeatures,testFeaturesS,testFeaturesW))
    print('Test Features Size:', alltestFeatures.shape)

    testpcaFeatures=pca.transform(alltestFeatures)
    print('\n-- Classification based on Distances --')
    nObsTest,nFea=alltestFeatures.shape
    for i in range(nObsTest):
        x=alltestFeatures[i]
        dists=[distance(x,centroids[0]),distance(x,centroids[1]),distance(x,centroids[2])]
        ndists=dists/np.sum(dists)
        testClass=np.argsort(dists)[0]
        
        print('Obs: {:2}: Normalized Distances to Centroids: [{:.4f},{:.4f},{:.4f}] -> Classification: {} -> {}'.format(i, *ndists, testClass, traffic_classes[testClass]))

    ## -- 12 -- #

    print('\n-- Classification based on Multivariate PDF (PCA Features) --')
    means={}
    for c in range(3):
        pClass=(t_classes==c).flatten()
        means.update({c:np.mean(pca_features[pClass,:],axis=0)})
    #print(means)

    covs={}
    for c in range(3):
        pClass=(t_classes==c).flatten()
        covs.update({c:np.cov(pca_features[pClass,:],rowvar=0)})
    #print(covs)

    testpcaFeatures=pca.transform(alltestFeatures)  #uses pca fitted above, only transforms test data
    print(testpcaFeatures)
    nObsTest,nFea=testpcaFeatures.shape
    for i in range(nObsTest):
        x=testpcaFeatures[i,:]
        probs=np.array([multivariate_normal.pdf(x,means[0],covs[0]),multivariate_normal.pdf(x,means[1],covs[1]),multivariate_normal.pdf(x,means[2],covs[2])])
        testClass=np.argsort(probs)[-1]
        
        print('Obs: {:2}: Probabilities: [{:.4e},{:.4e},{:.4e}] -> Classification: {} -> {}'.format(i, *probs, testClass, traffic_classes[testClass]))

    ## -- 13 -- ##
    scaler=StandardScaler()
    NormAllFeatures=scaler.fit_transform(all_features)

    NormAllTestFeatures=scaler.fit_transform(alltestFeatures)

    pca = PCA(n_components=3, svd_solver='full')
    NormPcaFeatures = pca.fit(NormAllFeatures).transform(NormAllFeatures)

    NormTestPcaFeatures = pca.fit(NormAllTestFeatures).transform(NormAllTestFeatures)

    ##

    print('\n-- Classification based on Clustering (Kmeans) --')
    
    #K-means assuming 3 clusters
    centroids=np.array([])
    for c in range(3):
        pClass=(t_classes==c).flatten()
        centroids=np.append(centroids,np.mean(NormPcaFeatures[pClass,:],axis=0))
    centroids=centroids.reshape((3,3))
    print('PCA (pca_features) Centroids:\n',centroids)

    kmeans = KMeans(init=centroids, n_clusters=3)
    kmeans.fit(NormPcaFeatures)
    labels=kmeans.labels_
    print('Labels:',labels)

    #Determines and quantifies the presence of each original class observation in each cluster
    KMclass=np.zeros((3,3))
    for cluster in range(3):
        p=(labels==cluster)
        aux=t_classes[p]
        for c in range(3):
            KMclass[cluster,c]=np.sum(aux==c)

    probKMclass=KMclass/np.sum(KMclass,axis=1)[:,np.newaxis]
    print(probKMclass)
    nObsTest,nFea=NormTestPcaFeatures.shape
    for i in range(nObsTest):
        x=NormTestPcaFeatures[i,:].reshape((1,nFea))
        label=kmeans.predict(x)
        testClass=100*probKMclass[label,:].flatten()
        print('Obs: {:2}: Probabilities beeing in each class: [{:.2f}%,{:.2f}%,{:.2f}%]'.format(i,*testClass))



    ## -- 14 -- ##
    
    #DBSCAN assuming a neighborhood maximum distance of 1e11
    dbscan = DBSCAN(eps=10000)
    dbscan.fit(pca_features)
    labels=dbscan.labels_
    print('Labels:',labels)

    ## -- 15 -- #
    
    print('\n-- Classification based on Support Vector Machines --')
    svc = svm.SVC(kernel='linear').fit(NormAllFeatures, t_classes)
    rbf_svc = svm.SVC(kernel='rbf').fit(NormAllFeatures, t_classes)
    poly_svc = svm.SVC(kernel='poly',degree=2).fit(NormAllFeatures, t_classes)
    lin_svc = svm.LinearSVC().fit(NormAllFeatures, t_classes)

    L1=svc.predict(NormAllTestFeatures)
    print('class (from test PCA features with SVC):',L1)
    L2=rbf_svc.predict(NormAllTestFeatures)
    print('class (from test PCA features with Kernel RBF):',L2)
    L3=poly_svc.predict(NormAllTestFeatures)
    print('class (from test PCA features with Kernel poly):',L3)
    L4=lin_svc.predict(NormAllTestFeatures)
    print('class (from test PCA features with Linear SVC):',L4)
    print('\n')

    nObsTest,nFea=NormAllTestFeatures.shape
    for i in range(nObsTest):
        print('Obs: {:2}: SVC->{} | Kernel RBF->{} | Kernel Poly->{} | LinearSVC->{}'.format(i, traffic_classes[L1[i]], traffic_classes[L2[i]], traffic_classes[L3[i]], traffic_classes[L4[i]]))

    ## -- 16 -- #
    print('\n-- Classification based on Support Vector Machines  (PCA Features) --')
    svc = svm.SVC(kernel='linear').fit(NormPcaFeatures, t_classes)
    rbf_svc = svm.SVC(kernel='rbf').fit(NormPcaFeatures, t_classes)
    poly_svc = svm.SVC(kernel='poly',degree=2).fit(NormPcaFeatures, t_classes)
    lin_svc = svm.LinearSVC().fit(NormPcaFeatures, t_classes)

    L1=svc.predict(NormTestPcaFeatures)
    print('class (from test PCA features with SVC):',L1)
    L2=rbf_svc.predict(NormTestPcaFeatures)
    print('class (from test PCA features with Kernel RBF):',L2)
    L3=poly_svc.predict(NormTestPcaFeatures)
    print('class (from test PCA features with Kernel poly):',L3)
    L4=lin_svc.predict(NormTestPcaFeatures)
    print('class (from test PCA features with Linear SVC):',L4)
    print('\n')

    nObsTest,nFea=NormTestPcaFeatures.shape
    for i in range(nObsTest):
        print('Obs: {:2}: SVC->{} | Kernel RBF->{} | Kernel Poly->{} | LinearSVC->{}'.format(i, traffic_classes[L1[i]], traffic_classes[L2[i]], traffic_classes[L3[i]], traffic_classes[L4[i]]))

    ## -- 17 -- ##
    
    print('\n-- Classification based on Neural Networks --')

    alpha=1
    max_iter=100000
    clf = MLPClassifier(solver='lbfgs',alpha=alpha,hidden_layer_sizes=(100,),max_iter=max_iter)
    clf.fit(NormPcaFeatures, t_classes)
    LT=clf.predict(NormTestPcaFeatures) 
    print('class (from test PCA):',LT)

    nObsTest,nFea=NormTestPcaFeatures.shape
    for i in range(nObsTest):
        print('Obs: {:2}: Classification->{}'.format(i, traffic_classes[LT[i]]))


if __name__ == '__main__':
    main()