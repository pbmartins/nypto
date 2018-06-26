![Nypto Logo](https://i.imgur.com/FrCFsYj.png)

**Nypto** is a network monitoring solution that detects cryptomining activities that may or not be hidden on the local machines. It is intented to be running in strategic places (Linux appliances on access switches mirroring ports as depicted in the example architecture below) and its impact on the network is negligible.

![Nypto Architecture](https://i.imgur.com/2T4lqTE.png)

# Models

**Nypto** is divided in two models:

* 🔸 *Offline model* (master branch): this model is heavily optimized to work on the given datasets and it is not prepared to work on a live scenario, when packets are captured and classified on the go. However, it has a wider range on scenarios, which particularly included traffic mixes of different classes, making the classification results less precise.
* 🔹 *Live-filtering model* (live-filtering branch): this model is simpler than the previous one, not including any type of mixed classes, and is prepared for live capturing and classification of packets.

# Datasets

Many traces of various traffic classes were obtained to make this models realistic in today's internet reality:

* YouTube 🔸🔹
* Netflix 🔸🔹
* Browsing 🔸🔹
* Social Networking 🔸🔹
* Email 🔸🔹
* VPN tunneling (Netflix, YouTube, CPU Mining 2&4 threads) 🔸🔹
* CPU Mining (2&4 threads mining Neoscrypt) 🔸🔹
* GPU Mining (EquiHash - 60% usage on GTX 1070 and 85%-100% on GTX 1080Ti) 🔸🔹
* Normal traffic mixes 🔸
    - Browsing & Netflix
    - Browsing & Social Networking
    - Browsing & Youtube
    - Netflix & Social Networking
    - Netflix & YouTube
    - Social Networking & YouTube
* Mining traffic mixes 🔸
    - CPU Mining (2&4 threads mining Neoscrypt) & Browsing
    - CPU Mining (2&4 threads mining Neoscrypt) & Netflix
    - CPU Mining (2&4 threads mining Neoscrypt) & Social Networking
    - CPU Mining (2&4 threads mining Neoscrypt) & YouTube
    - GPU Mining (EquiHash - 60% usage on GTX 1070 and 85%-100% on GTX 1080Ti) & Browsing
    - GPU Mining (EquiHash - 60% usage on GTX 1070 and 85%-100% on GTX 1080Ti) & Netflix
    - GPU Mining (EquiHash - 60% usage on GTX 1070 and 85%-100% on GTX 1080Ti) & Social Networking
    - GPU Mining (EquiHash - 60% usage on GTX 1070 and 85%-100% on GTX 1080Ti) & YouTube

# Files

* `parse_packets.py`: Obtains packet counts (number of download/upload bytes and packets) from Wireshark captures and writes them to a file;
* `generate_merge_datasets.py`: Generate new dataset, resultant from the merge of a set of given datasets;
* `scalogram.py`: Returns Scalograms/Wavelets scales values from a given time window;
* `profiling.py`: Breaks the datasets into multiple windows (sliding windows), obtains its features and return the conjunction of all datasets features;
* `classification.py`: Classifies windows using machine learning algorithms;
* `filtering.py`: Live capture and filtering of traffic.

# Profiling

Each window has a set of features that were extracted:
* Upload/download packet and bytes count average;
* Upload/download packet and bytes count median;
* Upload/download packet and bytes count standard deviation;
* Upload/download packet and bytes count 75, 90 and 95 percentils;
* Upload/download packet and bytes silent periods average;
* Upload/download packet and bytes silent periods variance;
* Upload/download packet and bytes scalograms (scales 2 and 4).

These features are also normalized and processed by PCA.

# Classification

## "Offline" model

| Classification techniques | Nº Classes | Window size  (slide) | Window Aggr. (threshold) | True positives | False negatives | False positives | True negatives | Precision | Recall | Accuracy |
|:--------------------------------------------------------:|:----------:|:--------------------:|:------------------------:|:--------------:|:---------------:|:---------------:|:--------------:|:---------:|:------:|:--------:|
| Random forests (global model) /  SVM SVC (silence model) | 31 | 6 min (20 s) | 40 (0.60) | 1702 | 1121 | 111 | 10422 | 0.9388 | 0.6029 | 0.9078 |
| SVM SVC (global model) | 39 | 2 min (20 s) | 1 | 1201 | 1455 | 704 | 11469 | 0.6304 | 0.4522 | 0.8544 |
| SVM SVC (global model) | 39 | 2 min (20 s) | 40 (0.55) | 1269 | 1452 | 636 | 11472 | 0.6661 | 0.4664 | 0.8592 |
| SVM SVC (global model) | 39 | 6 min (20 s) | 40 (0.60) | 1189 | 845 | 624 | 11863 | 0.6558 | 0.5846 | 0.8988 |

## Live filtering model

| Classification techniques | Nº Classes | Window size  (slide) | Window Aggr. (threshold) | True positives | False negatives | False positives | True negatives | Precision | Recall | Accuracy |
|:-------------------------:|:----------:|:--------------------:|:------------------------:|:--------------:|:---------------:|:---------------:|:--------------:|:---------:|:------:|:--------:|
| SVM Linear SVC (binary classsification) | 2 | 2 min (20 s) | 50 (0.55) | 714 | 98 | 297 | 9704 | 0.7062 | 0.8793 | 0.9635 |
| SVM SVC (global model) | 13 | 6 min (20 s) | 70 (0.55) | 967 | 127 | 0 | 9619 | 1.0 | 0.8839 | 0.9881 |

### Binary classification Confusion Matrix

```
    0       1
0 714.0   297.0 
1  98.0  9704.0
```

### 13 classes Confusion Matrix

```
        0     1     2     3     4     5     6     7     8     9    10    11    12    13 
  0 140.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 
  1   0.0 159.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 
  2   0.0   2.0 106.0  56.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 
  3   0.0   0.0   0.0 135.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 
  4   0.0   0.0   0.0  10.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 
  5   0.0   0.0   0.0  21.0   0.0 145.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 
  6   0.0   0.0   0.0   0.0   0.0 160.0  32.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 
  7   0.0   0.0   0.0   0.0   0.0  32.0   0.0  40.0  97.0   0.0   0.0   6.0   0.0   0.0 
  8   0.0   0.0   0.0   0.0   0.0   0.0   0.0  39.0  64.0   0.0   2.0   0.0  39.0   1.0 
  9   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   2.0   0.0   0.0 935.0   5.0   0.0 
 10   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 1970.0   0.0   0.0 
 11   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  11.0   0.0 6129.0   0.0   2.0 
 12   0.0   0.0   0.0   0.0   0.0  31.0  54.0  38.0   0.0   0.0   0.0  59.0   0.0   7.0 
 13   0.0   0.0   0.0   0.0   0.0   0.0  10.0   5.0   0.0   1.0   0.0 167.0   0.0   0.0 
```


Diogo Ferreira & Pedro Martins - 2018

