import os
import datetime
import pyshark
import argparse

INFILE_PATH = 'miner.pcapng'
OUTFILE_PATH = 'datasets/mining_4t_nicehash.dat'
SAMPLE_DELTA = 0.5

LOCAL_IP = '192.168.1.158'
LOCAL_IPV6 = '2001:8a0:de41:5501:9af7:5d3d:ed53:73dd'
REMOTE_PORTS = [3341, 3333, 80, 443]

def save_to_file(delta, last_bytes_up, last_bytes_down,
        last_npkts_up, last_npkts_down, last_nsyns_up, last_nsyns_down):
        global OUTFILE_PATH

        # Save to file
        with open(OUTFILE_PATH, "a") as f:
            f.write("{} {} {} {} {} {}\n".format(last_bytes_up, last_bytes_down,
                last_npkts_up, last_npkts_down, last_nsyns_up, last_nsyns_down))
            for i in range(int(delta) - 1):
                f.write("0 0 0 0 0 0\n");


def process_packets(tcp_cap):
    global SAMPLE_DELTA
    
    last_timestamp = None
    last_bytes_up = 0
    last_bytes_down = 0
    last_npkts_up = 0
    last_npkts_down = 0
    last_nsyns_up = 0
    last_nsyns_down = 0

    for packet in tcp_cap:
        packet_type = -1
        if 'ipv6' in [l.layer_name for l in packet.layers]:
            ip = LOCAL_IPV6
            src = packet.ipv6.src
            dst = packet.ipv6.dst
        else:
            ip = LOCAL_IP
            src = packet.ip.src
            dst = packet.ip.dst

        if src == ip and int(packet.tcp.get_field('DstPort')) in REMOTE_PORTS:
            packet_type = 0
        elif int(packet.tcp.get_field('SrcPort')) in REMOTE_PORTS and dst == ip:
            packet_type = 1

        if packet_type == 0:
            if last_timestamp is None:
                last_timestamp = float(packet.sniff_timestamp)
                last_bytes_down += int(packet.tcp.get_field('Len'))
                last_npkts_down += 1
                last_nsyns_down += 1 if packet.tcp.get_field(
                        'Flags').main_field.hex_value & 18 == 18 else 0
            else:
                time_delta = float(packet.sniff_timestamp) - last_timestamp
                if time_delta > SAMPLE_DELTA:
                    save_to_file(time_delta, last_bytes_up, last_bytes_down,
                            last_npkts_up, last_npkts_down, last_nsyns_up, 
                            last_nsyns_down)
                    last_timestamp = float(packet.sniff_timestamp)
                    last_bytes_down = int(packet.tcp.get_field('Len'))
                    last_npkts_down = 1
                    last_nsyns_down = 1 if packet.tcp.get_field(
                            'Flags').main_field.hex_value & 18 == 18 else 0
                else:
                    last_bytes_down += int(packet.tcp.get_field('Len'))
                    last_npkts_down += 1
                    last_nsyns_down += 1 if packet.tcp.get_field(
                            'Flags').main_field.hex_value & 18 == 18 else 0
        elif packet_type == 1:
            if last_timestamp is None:
                last_timestamp = float(packet.sniff_timestamp)
                last_bytes_up += int(packet.tcp.get_field('Len'))
                last_npkts_up += 1
                last_nsyns_up += 1 if packet.tcp.get_field(
                        'Flags').main_field.hex_value & 18 == 18 else 0
            else:
                time_delta = float(packet.sniff_timestamp) - last_timestamp
                if time_delta > SAMPLE_DELTA:
                    save_to_file(time_delta, last_bytes_up, last_bytes_down,
                            last_npkts_up, last_npkts_down, last_nsyns_up, 
                            last_nsyns_down)
                    last_timestamp = float(packet.sniff_timestamp)
                    last_bytes_up = int(packet.tcp.get_field('Len'))
                    last_npkts_up = 1
                    last_nsyns_up = 1 if packet.tcp.get_field(
                            'Flags').main_field.hex_value & 18 == 18 else 0
                else:
                    last_bytes_up += int(packet.tcp.get_field('Len'))
                    last_npkts_up += 1
                    last_nsyns_up += 1 if packet.tcp.get_field(
                            'Flags').main_field.hex_value & 18 == 18 else 0


    save_to_file(0, last_bytes_up, last_bytes_down, last_npkts_up, 
            last_npkts_down, last_nsyns_up, last_nsyns_down)


def main():
    global INFILE_PATH
    global OUTFILE_PATH
    global SAMPLE_DELTA
    global LOCAL_IP
    global LOCAL_IPV6

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='?',
            help='input capture file')
    parser.add_argument('-o', '--output', nargs='?',
            help='output processed file')
    parser.add_argument('-w', '--sampwindow', nargs='?',
            help='sampling interval (default 0.5s)')
    parser.add_argument('-4', '--ipv4', nargs='?',
            help='IPv4 of the host machine')
    parser.add_argument('-6', '--ipv6', nargs='?',
            help='IPv6 of the host machine')
    args = parser.parse_args()

    INFILE_PATH = args.input if args.input is not None else INFILE_PATH
    OUTFILE_PATH = args.output if args.output is not None else OUTFILE_PATH
    SAMPLE_DELTA = args.sampwindow if args.sampwindow is not None else SAMPLE_DELTA
    LOCAL_IP = args.ipv4 if args.ipv4 is not None else LOCAL_IP
    LOCAL_IPV6 = args.ipv6 if args.ipv6 is not None else LOCAL_IPV6

    if os.path.exists(OUTFILE_PATH):
        if input('Write over file? [y/N]') == 'y':
            os.remove(OUTFILE_PATH)
        else:
            exit()

    tcp_cap = pyshark.FileCapture(
            INFILE_PATH, display_filter='tcp', keep_packets=False)

    process_packets(tcp_cap)


if __name__ == '__main__':
    main()
