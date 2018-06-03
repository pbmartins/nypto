import os
import sys
import argparse
import pyshark
import numpy as np
from netaddr import IPNetwork, IPAddress, IPSet
from itertools import groupby
import classification

N_PACKETS = 0
OUTFILE_PATH = 'samples/'
SAMPLE_DELTA = 0.5
WINDOW_DELTA = 240
SRC_IP_ALLOCATE = 20
DST_IP_ALLOCATE = 100
TCP_PORT_ALLOCATE = 20
CLIENT_NETS_SET = None
N_FEATURES = 7
MAX_CLASSIFICATIONS = 5
TRAFFIC_STATS = None
TRAFFIC_CLASSIFICATIONS = None
LOCAL_IPS = {}
REMOTE_IPS = {}
TCP_PORTS = {}


def add_new_src_ip():
    global LOCAL_IPS
    global SRC_IP_ALLOCATE
    global DST_IP_ALLOCATE
    global TCP_PORT_ALLOCATE
    global N_FEATURES
    global MAX_CLASSIFICATIONS
    global TRAFFIC_STATS
    global TRAFFIC_CLASSIFICATIONS

    if len(LOCAL_IPS) <= TRAFFIC_STATS.shape[0]:
        return

    # Pre allocate memory space
    remote_ips_size = TRAFFIC_STATS.shape[1]
    tcp_ports_size = TRAFFIC_STATS.shape[2]
    new_entry = np.zeros((IP_ALLOCATE, remote_ips_size, tcp_ports_size,
                          WINDOW_DELTA, N_FEATURES))
    TRAFFIC_STATS = np.vstack((TRAFFIC_STATS, new_entry))

    new_entry = np.zeros((IP_ALLOCATE, remote_ips_size, tcp_ports_size,
                          MAX_CLASSIFICATIONS))
    TRAFFIC_CLASSIFICATIONS = np.vstack((TRAFFIC_CLASSIFICATIONS, new_entry))


def add_new_dst_ip():
    global LOCAL_IPS
    global SRC_IP_ALLOCATE
    global DST_IP_ALLOCATE
    global TCP_PORT_ALLOCATE
    global N_FEATURES
    global MAX_CLASSIFICATIONS
    global TRAFFIC_STATS
    global TRAFFIC_CLASSIFICATIONS

    if len(REMOTE_IPS) <= TRAFFIC_STATS.shape[1]:
        return

    # Pre allocate memory space
    tcp_ports_size = TRAFFIC_STATS.shape[2]
    new_entry = np.zeros((DST_IP_ALLOCATE, tcp_ports_size,
                          WINDOW_DELTA, N_FEATURES))
    new_stats = np.array([np.vstack((TRAFFIC_STATS[0], new_entry))])

    new_entry_class = np.zeros((DST_IP_ALLOCATE, tcp_ports_size,
                          MAX_CLASSIFICATIONS))
    new_stats_class = \
        np.array([np.vstack((TRAFFIC_CLASSIFICATIONS[0], new_entry_class))])

    for ip in range(1, TRAFFIC_STATS.shape[0]):
        new_stats = np.vstack((
            new_stats, [np.vstack((TRAFFIC_STATS[ip], new_entry))]))
        new_stats_class = np.vstack((
            new_stats_class,
            [np.vstack((TRAFFIC_CLASSIFICATIONS[ip], new_entry_class))]))

    TRAFFIC_STATS = new_stats
    TRAFFIC_CLASSIFICATIONS = new_stats_class


def add_new_tcp_port():
    global LOCAL_IPS
    global N_FEATURES
    global MAX_CLASSIFICATIONS
    global TRAFFIC_STATS
    global TRAFFIC_CLASSIFICATIONS

    if len(TCP_PORTS) <= TRAFFIC_STATS.shape[1]:
        return

    # Pre allocate memory space
    new_entry = np.zeros((TCP_PORT_ALLOCATE, WINDOW_DELTA, N_FEATURES))
    new_stats = None

    new_entry_class = np.zeros((TCP_PORT_ALLOCATE, MAX_CLASSIFICATIONS))
    new_stats_class = None

    for src_ip in range(0, TRAFFIC_STATS.shape[0]):
        src_stats = np.array([np.vstack((TRAFFIC_STATS[src_ip][0], new_entry))])
        src_stats_class = np.array(
            [np.vstack((TRAFFIC_CLASSIFICATIONS[src_ip][0], new_entry_class))])

        for dst_ip in range(1, TRAFFIC_STATS.shape[1]):
            src_stats = np.vstack((
                src_stats, [np.vstack((TRAFFIC_STATS[src_ip][dst_ip], new_entry))]))
            src_stats_class = np.vstack((
                src_stats_class,
                [np.vstack((TRAFFIC_CLASSIFICATIONS[src_ip][dst_ip], new_entry_class))]))

        new_stats = np.array([src_stats]) if src_ip == 0 \
            else np.vstack((new_stats, [src_stats]))
        new_stats_class = np.array([src_stats_class]) if src_ip == 0 \
            else np.vstack((new_stats_class, [src_stats_class]))

    TRAFFIC_STATS = new_stats
    TRAFFIC_CLASSIFICATIONS = new_stats_class


def classify(local_ip, remote_ip, remote_port):
    global MAX_CLASSIFICATIONS
    global TRAFFIC_STATS
    global TRAFFIC_CLASSIFICATIONS

    src_idx = LOCAL_IPS[local_ip]
    dst_idx = REMOTE_IPS[remote_ip]
    port_idx = TCP_PORTS[remote_port]

    print(TRAFFIC_STATS[src_idx][dst_idx][port_idx])
    # CALL CLASSIFY
    # traffic_class = classify(...)
    traffic_class = 1

    class_idx = 0
    for idx in TRAFFIC_CLASSIFICATIONS[src_idx][dst_idx][port_idx]:
        if TRAFFIC_CLASSIFICATIONS[src_idx][dst_idx][port_idx][idx] == 0:
            class_idx = idx
            break

    TRAFFIC_CLASSIFICATIONS[src_idx][dst_idx][port_idx][class_idx] = traffic_class

    # Classify traffic based on historic
    if class_idx == MAX_CLASSIFICATIONS - 1:
        ordered_class = sorted(TRAFFIC_CLASSIFICATIONS[src_idx][dst_idx][port_idx])
        classes = {key: len(list(group)) for key, group in groupby(ordered_class)}
        traffic_class = max(classes)

        # If it is mining
        if traffic_class == 1:
            # Block in the firewall
            print("TCP flow with src IP {} and dst IP {} is running mining "
                  "on port {}".format(local_ip, remote_ip, remote_port))

            # Update classification matrix
            TRAFFIC_CLASSIFICATIONS[src_idx][dst_idx][port_idx] = np.zeros((1, MAX_CLASSIFICATIONS))
            return -1

    return 0


def pkt_callback(pkt):
    global CLIENT_NETS_SET
    global N_PACKETS
    global LOCAL_IPS
    global REMOTE_IPS
    global TCP_PORTS

    print(pkt)

    src_ip = IPAddress(pkt.ip.src)
    dst_ip = IPAddress(pkt.ip.dst)
    src_port = pkt.tcp.srcport
    dst_port = pkt.tcp.dstport

    # Verify if it's a valid IP prefix
    if src_ip in CLIENT_NETS_SET:
        local_ip = src_ip
        remote_ip = dst_ip
        remote_port = dst_port
        up_down = 0
    elif dst_ip in CLIENT_NETS_SET:
        local_ip = dst_ip
        remote_ip = src_ip
        remote_port = src_port
        up_down = 1
    else:
        return None

    if local_ip not in LOCAL_IPS:
        LOCAL_IPS[local_ip] = len(LOCAL_IPS)
        add_new_src_ip()

    if remote_ip not in REMOTE_IPS:
        REMOTE_IPS[remote_ip] = len(REMOTE_IPS)
        add_new_dst_ip()

    if remote_port not in TCP_PORTS:
        idx = None
        for port in TCP_PORTS:
            idx = TCP_PORTS[port] if TCP_PORTS[port] is None else None
        TCP_PORTS[remote_port] = len(TCP_PORTS) if idx is None else idx
        add_new_tcp_port()

    src_idx = LOCAL_IPS[local_ip]
    dst_idx = REMOTE_IPS[remote_ip]
    port_idx = TCP_PORTS[remote_port]
    info = TRAFFIC_STATS[src_idx][dst_idx][port_idx]
    N_PACKETS += 1

    time_delta = float(pkt.sniff_timestamp) - info[0][0] if info[0][0] != -1 \
        else float(pkt.sniff_timestamp)
    idx = 0 if time_delta == 0 else int(time_delta / SAMPLE_DELTA)

    if idx > WINDOW_DELTA:
        rtn = classify(local_ip, remote_ip, remote_port)
        TRAFFIC_STATS[src_idx][dst_idx][port_idx] = \
            np.zeros((WINDOW_DELTA, N_FEATURES))
        TRAFFIC_STATS[src_idx][dst_idx][port_idx][0][0] = rtn
        idx = 0
        info[idx][0] = int(pkt.sniff_timestamp)

    info[idx][1+up_down] += int(pkt.tcp.get_field('Len'))
    info[idx][3+up_down] += 1
    info[idx][5+up_down] += 1 if pkt.tcp.get_field(
        'Flags').main_field.hex_value & 18 == 18 else 0

    TRAFFIC_STATS[src_idx][dst_idx][port_idx] = info


def main():
    global CLIENT_NETS_SET
    global SAMPLE_DELTA
    global IP_ALLOCATE
    global TCP_PORT_ALLOCATE
    global TRAFFIC_STATS
    global TRAFFIC_CLASSIFICATIONS

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--interface', nargs='?',
                        required=True, help='capture interface')
    parser.add_argument('-c', '--cnet', nargs='+',
                        required=True, help='client network(s)')
    parser.add_argument('-w', '--sampwindow', nargs='?',
                        help='sampling interval (default: 0.5 s)')
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
    print('TCP filter active on {}'.format(net_interface))

    SAMPLE_DELTA = args.sampwindow if args.sampwindow is not None else SAMPLE_DELTA

    TRAFFIC_STATS = \
        np.zeros((SRC_IP_ALLOCATE, DST_IP_ALLOCATE, TCP_PORT_ALLOCATE,
                  WINDOW_DELTA, N_FEATURES))
    TRAFFIC_CLASSIFICATIONS = \
        np.zeros((SRC_IP_ALLOCATE, DST_IP_ALLOCATE, TCP_PORT_ALLOCATE,
                  MAX_CLASSIFICATIONS))

    try:
        capture = pyshark.LiveCapture(interface=net_interface, bpf_filter='tcp')
        capture.apply_on_packets(pkt_callback)
    except KeyboardInterrupt:
        print('\n{} packets captured! Done!\n'.format(N_PACKETS))
        print(0)


if __name__ == '__main__':
    main()
