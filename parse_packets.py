import datetime
import pyshark
import argparse

INFILE_PATH = 'miner.pcapng'
OUTFILE_PATH = 'mining_4t_nicehash.dat'
SAMPLE_DELTA = 0.5

LOCAL_IP = '192.168.1.158'
REMOTE_MINER_PORT = 3341

def save_to_file(delta, last_bytes_up, last_bytes_down,
        last_npkts_up, last_npkts_down):
        global OUTFILE_PATH

        # Save to file
        with open(OUTFILE_PATH, "a") as f:
            f.write("{} {} {} {}\n".format(last_bytes_up, last_bytes_down,
                last_npkts_up, last_npkts_down))
            for i in range(int(delta) - 1):
                f.write("0 0 0 0\n");


def process_packets(tcp_cap):
    global SAMPLE_DELTA
    
    last_timestamp = None
    last_bytes_up = 0
    last_bytes_down = 0
    last_npkts_up = 0
    last_npkts_down = 0

    for packet in tcp_cap:
        packet_type = -1
        if packet.ip.src == LOCAL_IP and \
                int(packet.tcp.get_field('DstPort')) == REMOTE_MINER_PORT:
            packet_type = 0
        elif int(packet.tcp.get_field('SrcPort')) == REMOTE_MINER_PORT and \
                packet.ip.dst == LOCAL_IP:
            packet_type = 1

        if packet_type == 0:
            if last_timestamp is None:
                last_timestamp = float(packet.sniff_timestamp)
                last_bytes_down += int(packet.length)
                last_npkts_down += 1
            else:
                time_delta = float(packet.sniff_timestamp) - last_timestamp
                if time_delta > SAMPLE_DELTA:
                    save_to_file(time_delta, last_bytes_up, last_bytes_down,
                            last_npkts_up, last_npkts_down)
                    last_timestamp = float(packet.sniff_timestamp)
                    last_bytes_down = int(packet.length)
                    last_npkts_down = 1
                else:
                    last_bytes_down += int(packet.length)
                    last_npkts_down += 1
        elif packet_type == 1:
            if last_timestamp is None:
                last_timestamp = float(packet.sniff_timestamp)
                last_bytes_up += int(packet.length)
                last_npkts_up += 1
            else:
                time_delta = float(packet.sniff_timestamp) - last_timestamp
                if time_delta > SAMPLE_DELTA:
                    save_to_file(time_delta, last_bytes_up, last_bytes_down,
                            last_npkts_up, last_npkts_down)
                    last_timestamp = float(packet.sniff_timestamp)
                    last_bytes_up = int(packet.length)
                    last_npkts_up = 1
                else:
                    last_bytes_up += int(packet.length)
                    last_npkts_up += 1


    save_to_file(0, last_bytes_up, last_bytes_down, last_npkts_up, last_npkts_down)


def main():
    global INFILE_PATH
    global OUTFILE_PATH
    global SAMPLE_DELTA

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='?',
            help='input capture file')
    parser.add_argument('-o', '--output', nargs='?',
            help='output processed file')
    parser.add_argument('-w', '--sampwindow', nargs='?',
            help='sampling interval (default 1s)')
    args = parser.parse_args()

    INFILE_PATH = args.input if args.input is not None else INFILE_PATH
    OUTFILE_PATH = args.output if args.output is not None else OUTFILE_PATH
    SAMPLE_DELTA = args.sampwindow if args.sampwindow is not None else SAMPLE_DELTA

    tcp_cap = pyshark.FileCapture(
            INFILE_PATH, display_filter='tcp', keep_packets=False)

    process_packets(tcp_cap)


if __name__ == '__main__':
    main()
