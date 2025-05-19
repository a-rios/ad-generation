import json
import pandas as pd
import argparse
from pathlib import Path
import os




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsons', metavar='PATH', type=str, help="Path to json files with ADs and start/end times.")
    parser.add_argument('--exclude', metavar='PATH', type=str, nargs="+", help="List of file names to exclude (test files)")
    parser.add_argument('--csv', metavar='PATH',type=str, help="csv with durations, ADs, ratios")
    args = parser.parse_args()


    columns = []
    df_stats = pd.DataFrame(columns=columns)

    for infile in os.listdir(args.jsons):
        if infile.endswith(".json") and infile not in args.exclude:
            name = Path(infile).stem
            print(name)
            df = pd.read_json(os.path.join(args.jsons,infile)).reset_index(drop=True)
            df['name'] = name

            df['start_s'] = pd.to_timedelta(df['start'])
            df['end_s'] = pd.to_timedelta(df['end'])
            df['duration'] = (df['end_s'] - df['start_s']).dt.total_seconds()


            df['word_count'] = df['source'].str.split().str.len()
            df['words_per_s'] = df['word_count'] / df['duration']

            df_stats = pd.concat([df_stats, df[['name', 'start', 'end', 'duration', 'word_count', 'words_per_s']]], axis=0)
            print(df_stats)



    df_stats.to_csv(args.csv, index=False)
