import os
import sys
import argparse
import pyshark
import numpy as np
from netaddr import IPNetwork, IPAddress, IPSet
import classification

N_PACKETS = 0
OUTFILE_PATH = 'samples/'
SAMPLE_DELTA = 0.5
WINDOW_DELTA = 240
IP_ALLOCATE = 20
PORT_ALLOCATE = 20
CLIENT_NETS_SET = None
N_FEATURES = 7
TRAFFIC_STATS = None
LOCAL_IPS = {}
TCP_PORTS = {}


def add_new_ip():
    global LOCAL_IPS
    global IP_ALLOCATE
    global PORT_ALLOCATE
    global N_FEATURES
    global TRAFFIC_STATS

    if len(LOCAL_IPS) <= TRAFFIC_STATS.shape[0]:
        return

    # Pre allocate memory space
    new_entry = np.zeros((IP_ALLOCATE, PORT_ALLOCATE, WINDOW_DELTA, N_FEATURES))
    TRAFFIC_STATS = np.vstack((TRAFFIC_STATS, new_entry))


def add_new_tcp_port():
    global LOCAL_IPS
    global N_FEATURES
    global TRAFFIC_STATS

    if len(TCP_PORTS) <= TRAFFIC_STATS.shape[1]:
        return

    # Pre allocate memory space
    new_entry = np.zeros((PORT_ALLOCATE, WINDOW_DELTA, N_FEATURES))
    new_stats = np.array([np.vstack((TRAFFIC_STATS[0], new_entry))])

    for ip in range(1, TRAFFIC_STATS.shape[0]):
        new_stats = np.vstack((
            new_stats, [np.vstack((TRAFFIC_STATS[ip], new_entry))]))

    TRAFFIC_STATS = new_stats


def save_samples(local_ip, remote_port, delta):
    global OUTFILE_PATH
    global LOCAL_IPS
    global TCP_PORTS

    ip_idx = LOCAL_IPS[local_ip]
    port_idx = TCP_PORTS[remote_port]
    filename = "{}-{}.dat".format(ip_idx, port_idx)
    info = [int(TRAFFIC_STATS[ip_idx][port_idx][i])
            for i in range(2, N_FEATURES)]
    n_samples = int(TRAFFIC_STATS[ip_idx][port_idx][1])

    with open(OUTFILE_PATH + filename, "a") as f:
        n_samples += 1
        f.write("{} {} {} {} {} {}\n".format(info[0], info[1], info[2], info[3],
                                             info[4], info[5], info[6]))

        diff_intervals = int(delta / SAMPLE_DELTA)
        diff_intervals = diff_intervals \
            if diff_intervals + n_samples <= WINDOW_DELTA \
            else WINDOW_DELTA - diff_intervals
        n_samples += diff_intervals

        for i in range(diff_intervals):
            f.write("0 0 0 0 0 0\n")

    return n_samples


def classify(local_ip, remote_port):
    ip_idx = LOCAL_IPS[local_ip]
    port_idx = TCP_PORTS[remote_port]

    print(TRAFFIC_STATS[ip_idx][port_idx])
    # CALL CLASSIFY


def pkt_callback(pkt):
    global CLIENT_NETS_SET
    global N_PACKETS
    global LOCAL_IPS
    global TCP_PORTS

    print(pkt)

    src_ip = IPAddress(pkt.ip.src)
    dst_ip = IPAddress(pkt.ip.dst)
    src_port = pkt.tcp.srcport
    dst_port = pkt.tcp.dstport

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
        add_new_ip()

    if remote_port not in TCP_PORTS:
        TCP_PORTS[remote_port] = len(TCP_PORTS)
        add_new_tcp_port()

    info = TRAFFIC_STATS[LOCAL_IPS[local_ip]][TCP_PORTS[remote_port]]
    N_PACKETS += 1

    time_delta = float(pkt.sniff_timestamp) - info[0][0]
    idx = 0 if time_delta == 0 else int(time_delta / SAMPLE_DELTA)

    if idx > WINDOW_DELTA:
        classify(local_ip, remote_port)
        TRAFFIC_STATS[LOCAL_IPS[local_ip]][TCP_PORTS[remote_port]] = \
            np.zeros((WINDOW_DELTA, N_FEATURES))
        idx = 0
        info[idx][0] = int(pkt.sniff_timestamp)

    info[idx][1+up_down] += int(pkt.tcp.get_field('Len'))
    info[idx][3+up_down] += 1
    info[idx][5+up_down] += 1 if pkt.tcp.get_field(
        'Flags').main_field.hex_value & 18 == 18 else 0

    TRAFFIC_STATS[LOCAL_IPS[local_ip]][TCP_PORTS[remote_port]] = info


def main():
    global CLIENT_NETS_SET
    global SAMPLE_DELTA
    global OUTFILE_PATH
    global IP_ALLOCATE
    global PORT_ALLOCATE
    global TRAFFIC_STATS

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--interface', nargs='?',
                        required=True, help='capture interface')
    parser.add_argument('-c', '--cnet', nargs='+',
                        required=True, help='client network(s)')
    parser.add_argument('-o', '--output', nargs='?',
                        help='temporary samples folder (default: samples/)')
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

    OUTFILE_PATH = args.output if args.output is not None else OUTFILE_PATH
    SAMPLE_DELTA = args.sampwindow if args.sampwindow is not None else SAMPLE_DELTA

    TRAFFIC_STATS = \
        np.zeros((IP_ALLOCATE, PORT_ALLOCATE, WINDOW_DELTA, N_FEATURES))

    #if not os.path.isdir(OUTFILE_PATH):
    #    print("Invalid samples directory")
    #    exit(1)

    try:
        capture = pyshark.LiveCapture(interface=net_interface, bpf_filter='tcp')
        capture.apply_on_packets(pkt_callback)
    except KeyboardInterrupt:
        print('\n{} packets captured! Done!\n'.format(N_PACKETS))
        print(0)


if __name__ == '__main__':
    main()
