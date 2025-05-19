import json
import pandas as pd
import argparse
from pathlib import Path

# imdb ids:
# DE:
# Zuercher Tagebuch tt14745616, no cast on imdb!
# Moskau einfach tt10949014
# Baghdad in my shadow tt6864088

# IT:
# Io sono Babbo Natale tt12617312
# Quanto basta tt7117552
# La tentazione di esistere tt24248964


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_json', metavar='PATH', type=str, help="input json with ADs and start/end times")
    parser.add_argument('--in_csv', metavar='PATH',type=str, help="input csv from last annotation step in shot-by-shot")
    parser.add_argument('--out_csv', metavar='PATH',type=str, help="output csv for label studio")
    args = parser.parse_args()
    # text,start,end,movie,scaled_start,scaled_end,cmd_filename,imdbid,movie_title,cmd_clip_idx

    df = pd.read_json(args.json).reset_index(drop=True)

    with open(args.in_json, 'r') as f:
        data = json.load(f)


