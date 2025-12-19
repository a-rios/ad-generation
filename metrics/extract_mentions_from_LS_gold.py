import os
import sys
import json
import ast
import argparse
import pandas as pd

def main(args):

    # Read annotations produced by shot-by-shot's preprocessing
    df = pd.read_csv(args.LS_csv)

    mentions = df['label']

    with open(args.out_csv, 'w') as o:

        for mention in mentions:
            if pd.isna(mention):
                o.write(f"[]\n")
            else:
                mention_list = json.loads(mention)
                # in LS export, labels are a list with 1 element, change to just string
                for entity in mention_list:
                    label =  entity['labels'][0]
                    del entity['labels']
                    entity['label'] = label
                o.write(f"{mention_list}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--LS_csv', default=None, type=str, help='csv with output ADs')
    parser.add_argument('--out_csv', default=None, type=str, help='output csv')

    args = parser.parse_args()

    main(args)
