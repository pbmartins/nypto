import os
import argparse
import numpy as np
from functools import reduce


def parse_packets(cap_files, output_path):
    metrics = [np.loadtxt(f) for f in cap_files]
    min_dim = min([m.shape[0] for m in metrics])
    sum_func = lambda a, b: a + b
    metrics = reduce(sum_func, [m[:min_dim] for m in metrics])
    np.savetxt(output_path, metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='+',
                        required=True, help='input capture file(s)')
    parser.add_argument('-o', '--output', nargs='?',
                        required=True, help='output processed file')
    args = parser.parse_args()

    for f in args.input:
        if not os.path.exists(f):
            print("ERROR: Invalid input file!")
            exit(1)

    if os.path.exists(args.output):
        if input('Write over file? [y/N] ') == 'y':
            os.remove(args.output)
        else:
            exit()

    parse_packets(args.input, args.output)


if __name__ == '__main__':
    main()
