import sys
import os
import json
import srt
import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--srt', type=str, metavar='PATH', help='input srt file')
    parser.add_argument('--json', type=str, help='output path for json')
    return parser.parse_args()


def get_srt_info(srt_file):
    try:
        with open(srt_file, 'r', encoding="utf-8") as f:
            f = f.read()
    except UnicodeDecodeError:
        with open(srt_file, 'r', encoding="cp1252") as f:
            try:
                f = f.read()
            except UnicodeDecodeError:
                raise ValueError(f"Could not parse {srt_file}")

    return list(srt.parse(f))


def main():

    args = parse_args()
    output = []

    for srt in get_srt_info(args.srt):
        ad = {"index": srt.index, "start": srt.start, "end": srt.end, "source": srt.content}
        output.append(ad)

    with open(args.json, "w") as f:
        json.dump(output, f, indent=4, default=str)


if __name__ == "__main__":
    main()
