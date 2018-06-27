import os
import sys
import argparse
import pyshark
import numpy as np
from netaddr import IPNetwork, IPAddress, IPSet
from profiling import extract_live_features, normalize_live_features
from classification import classify_live_data

N_PACKETS = 0
OUTFILE_PATH = 'samples/'
SAMPLE_DELTA = 0.5
WINDOW_SIZE = 240
N_WINDOWS = 5
WINDOW_DELTA = WINDOW_SIZE * N_WINDOWS
SRC_IP_ALLOCATE = 40
TCP_PORT_ALLOCATE = 100
CLIENT_NETS_SET = None
N_FEATURES = 5
TRAFFIC_STATS = None
LOCAL_IPS = {}
TCP_PORTS = {}
BASE_TIMESTAMP = None
MINING_THRESHOLD = 0.7


def add_new_src_ip():
    global LOCAL_IPS
    global SRC_IP_ALLOCATE
    global TCP_PORT_ALLOCATE
    global N_FEATURES
    global TRAFFIC_STATS

    if len(LOCAL_IPS) < TRAFFIC_STATS.shape[0]:
        return

    # Pre allocate memory space
    tcp_ports_size = TRAFFIC_STATS.shape[1]
    new_entry = np.zeros((SRC_IP_ALLOCATE, tcp_ports_size, WINDOW_DELTA, N_FEATURES))
    TRAFFIC_STATS = np.vstack((TRAFFIC_STATS, new_entry))


def add_new_tcp_port():
    global LOCAL_IPS
    global TCP_PORT_ALLOCATE
    global N_FEATURES
    global TRAFFIC_STATS

    if len(TCP_PORTS) < TRAFFIC_STATS.shape[1]:
        return

    # Pre allocate memory space
    new_entry = np.zeros((TCP_PORT_ALLOCATE, WINDOW_DELTA, N_FEATURES))
    new_stats = np.array([np.vstack((TRAFFIC_STATS[0], new_entry))])

    for ip in range(1, TRAFFIC_STATS.shape[0]):
        new_stats = np.vstack((
            new_stats, [np.vstack((TRAFFIC_STATS[ip], new_entry))]))

    TRAFFIC_STATS = new_stats


def classify(local_ip, remote_port):
    global TRAFFIC_STATS
    global MINING_THRESHOLD

    src_idx = LOCAL_IPS[local_ip]
    port_idx = TCP_PORTS[remote_port]

    # Traffic profiling
    dataset = TRAFFIC_STATS[src_idx][port_idx][:, 1:]
    f, fs, fw = extract_live_features(dataset)
    all_features = np.hstack((f, fs, fw))

    # Less than 3 valid windows, cannot extract features with PCA
    if all_features.shape[0] < 3:
        print("No features could be extracted from TCP flow with src IP {} "
              "on port {}.".format(local_ip, remote_port))
        return -1

    norm_pca_features = normalize_live_features(all_features)

    # Traffic classification
    classes = classify_live_data(norm_pca_features)
    if classes['min'] >= MINING_THRESHOLD:
        # Block in the firewall
        print("TCP flow with src IP {} is running mining "
              "on port {} with {}% accuracy".format(
            local_ip, remote_port, classes['min']*100))

        return -1

    print("TCP flow with src IP {} is NOT mining "
          "on port {} with {}% accuracy".format(
        local_ip, remote_port, classes['nmin']*100))
    
    return 0


def pkt_callback(pkt):
    global CLIENT_NETS_SET
    global N_PACKETS
    global LOCAL_IPS
    global TCP_PORTS
    global BASE_TIMESTAMP

    if 'ipv6' in [l.layer_name for l in pkt.layers]:
        src_ip = IPAddress(pkt.ipv6.src)
        dst_ip = IPAddress(pkt.ipv6.dst)
        src_port = pkt.tcp.srcport
        dst_port = pkt.tcp.dstport
        size = int(pkt.ipv6.plen)
    else:
        src_ip = IPAddress(pkt.ip.src)
        dst_ip = IPAddress(pkt.ip.dst)
        src_port = pkt.tcp.srcport
        dst_port = pkt.tcp.dstport
        size = int(pkt.ip.get_field('Len'))

    if src_ip == IPAddress('94.63.100.39') or dst_ip == IPAddress('94.63.100.39'):
        return

    # Verify if it's a valid IP prefix
    if src_ip in CLIENT_NETS_SET:
        local_ip = src_ip
        remote_port = dst_port
        up_down = 0
    elif dst_ip in CLIENT_NETS_SET:
        local_ip = dst_ip
        remote_port = src_port
        up_down = 1
    else:
        return None

    if local_ip not in LOCAL_IPS:
        LOCAL_IPS[local_ip] = len(LOCAL_IPS)
        add_new_src_ip()

    if remote_port not in TCP_PORTS:
        idx = None
        for port in TCP_PORTS:
            idx = TCP_PORTS[port] if TCP_PORTS[port] is None else None
        TCP_PORTS[remote_port] = len(TCP_PORTS) if idx is None else idx
        add_new_tcp_port()

    src_idx = LOCAL_IPS[local_ip]
    port_idx = TCP_PORTS[remote_port]
    info = TRAFFIC_STATS[src_idx][port_idx]
    N_PACKETS += 1

    BASE_TIMESTAMP = float(pkt.sniff_timestamp) if BASE_TIMESTAMP is None \
            else BASE_TIMESTAMP

    timestamp = float(pkt.sniff_timestamp) - BASE_TIMESTAMP
    time_delta = timestamp - info[0][0] if info[0][0] != -1 else timestamp
    idx = 0 if time_delta == 0 else int(time_delta / SAMPLE_DELTA)

    if idx >= WINDOW_DELTA:
        rtn = classify(local_ip, remote_port)
        TRAFFIC_STATS[src_idx][port_idx] = np.zeros((WINDOW_DELTA, N_FEATURES))
        TRAFFIC_STATS[src_idx][port_idx][0][0] = rtn
        idx = 0
        info[idx][0] = timestamp

    info[idx][1+up_down] += size
    info[idx][3+up_down] += 1

    TRAFFIC_STATS[src_idx][port_idx] = info


def main():
    global CLIENT_NETS_SET
    global SAMPLE_DELTA
    global SRC_IP_ALLOCATE
    global TCP_PORT_ALLOCATE
    global TRAFFIC_STATS
    global MINING_THRESHOLD

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--interface', nargs='?',
                        required=True, help='capture interface')
    parser.add_argument('-c', '--cnet', nargs='+',
                        required=True, help='client network(s)')
    parser.add_argument('-m', '--miningthreshold', nargs='?',
                        help='mining detection threshold (default: 0.50)')
    args = parser.parse_args()

    client_networks = None
    try:
        client_networks = [IPNetwork(n) for n in args.cnet]
    except:
        print('Invalid valid network prefix')

    if len(client_networks) == 0:
        print("No valid client network prefixes.")
        sys.exit()

    CLIENT_NETS_SET = IPSet(client_networks)

    net_interface = args.interface
    print('TCP filter active on {} applied to the following '
            'networks: {}'.format(net_interface, CLIENT_NETS_SET))

    MINING_THRESHOLD = args.miningthreshold if args.miningthreshold is not None \
        else MINING_THRESHOLD

    TRAFFIC_STATS = \
        np.zeros((SRC_IP_ALLOCATE, TCP_PORT_ALLOCATE, WINDOW_DELTA, N_FEATURES))

    try:
        capture = pyshark.LiveCapture(interface=net_interface, bpf_filter='tcp')
        capture.apply_on_packets(pkt_callback)
    except KeyboardInterrupt:
        print('\n{} packets captured! Done!\n'.format(N_PACKETS))
        exit()


if __name__ == '__main__':
    main()
