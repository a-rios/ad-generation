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
    parser.add_argument('--json', metavar='PATH', type=str, help="input json with ADs and start/end times")
    parser.add_argument('--csv', metavar='PATH',type=str, help="output csv use in shot-by-shot")
    args = parser.parse_args()
    # text,start,end,movie,scaled_start,scaled_end,cmd_filename,imdbid,movie_title,cmd_clip_idx

    imdb = {
            "Zuercher_Tagebuch" : "tt14745616",
            "Moskau_einfach" : "tt10949014",
            "Baghdad_in_my_Shadow" : "tt6864088",
            "Io_Sono_Babbo_Natale" : "tt12617312",
            "Quanto_basta" : "tt7117552",
            "La_Tentazione_Di_Esistere" : "tt24248964",
            "Wanda_mein_Wunder" : "tt10152722",
            "Body_of_Truth" : "tt11382952",
            "Beautiful_Minds" : "tt13553662"
        }

    name = Path(args.json).stem
    df = pd.read_json(args.json).reset_index(drop=True)

    # Keep only specific columns
    df = df[['source', 'start', 'end', 'index']]  # replace with the fields you want
    df.rename(columns={'source': 'text'}, inplace=True)
    df.rename(columns={'index': 'AD_idx'}, inplace=True)
    df['movie_title'] = name
    df['start'] = pd.to_timedelta(df['start']).dt.total_seconds()
    df['end'] = pd.to_timedelta(df['end']).dt.total_seconds()
    # remove newlines
    df['text'] = df['text'].str.replace(r'[\r\n]+', ' ', regex=True)
    df['text'] = df['text'].str.strip()
    df['imdbid'] = imdb[name]
    print(df)

    df.to_csv(args.csv, index=False)
