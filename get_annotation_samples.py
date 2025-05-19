import json
import pandas as pd
import argparse
from pathlib import Path
import os
import random
import datetime

random.seed(42)



def get_same_samples(df, sampled_list, source):

    matched_groups = []
    df_lookup = df.set_index(['name', 'anno_idx'])

    for group in sampled_list:
        keys = [(row['name'], row['anno_idx']) for _, row in group.iterrows()]

        try:
            matched_rows = df_lookup.loc[keys].reset_index()
            matched_rows['source'] = source
            matched_groups.append(matched_rows)
        except KeyError:
            print(f"Did not find {keys} in df for {source}")
            exit(1)

    return matched_groups

def sample_consecutive_groups(df, group_size, n_groups, model):

    total_rows = len(df)
    max_full_groups = total_rows // group_size

    # non-overlapping group start indices
    group_starts = [i * group_size for i in range(max_full_groups)]

    # sample group start indices
    sampled_starts = random.sample(group_starts, min(n_groups, len(group_starts)))

    groups = []
    for start in sampled_starts:
        group = df.iloc[start:start + group_size].copy()
        group["source"] = model
        groups.append(group)

    return groups


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llama_csvs', metavar='PATH', type=str, nargs="+", help="List of stage2 llama3 csvs with generated ADs")
    parser.add_argument('--gpt_csvs', metavar='PATH', type=str, nargs="+", help="List of stage2 gpt4 csvs with generated ADs")
    parser.add_argument('--gen_sample_groups_per_model', type=int, help="number of generated AD samples")
    parser.add_argument('--gt_sample_groups', type=int, help="number of ground truth samples")
    parser.add_argument('--consecutive', type=int, help="sample in consecutive groups of N ads")
    parser.add_argument('--models_same_ads', action="store_true", help="sample the same ADs for both models (default: different ADs)")
    parser.add_argument('--gt_same_ads', action="store_true", help="sample the same ADs for ground truth (only if also --models_same_ads) ")
    parser.add_argument('--out_json', metavar='PATH',type=str, help="output json (format for label studio")
    args = parser.parse_args()


    ## label studio json
    # # [
    # #     {
    # #         "video": "http://localhost:8081/219_0-24-49.125000_0-24-51.625000.mp4",
    # #         "text": "Es reicht. Ich brauche den Computer"
    # #     }
    # # ]

    columns = []
    df_llama = pd.DataFrame(columns=columns)
    df_gpt = pd.DataFrame(columns=columns)
    for in_csv in args.llama_csvs:
        name = Path(in_csv).parent.stem
        df = pd.read_csv(in_csv)
        df['name'] = name
        df['model_name'] = "llama3"
        df_llama = pd.concat([df_llama, df[15:-15]], axis=0) # exclude first + last 15 ADs to avoid credits

    for in_csv in args.gpt_csvs:
        name = Path(in_csv).parent.stem
        df = pd.read_csv(in_csv)
        df['name'] = name
        df['model_name'] = "gpt4"
        df_gpt = pd.concat([df_gpt, df[15:-15]], axis=0) # exclude first + last 15 ADs to avoid credits

    if args.models_same_ads:
        llama3_samples = sample_consecutive_groups(df_llama, args.consecutive, args.gen_sample_groups_per_model, "llama3")
        gpt4_samples = get_same_samples(df_gpt, llama3_samples, "gpt4")

    else:
        llama3_samples = sample_consecutive_groups(df_llama, args.consecutive, args.gen_sample_groups_per_model, "llama3")
        gpt4_samples = sample_consecutive_groups(df_gpt, args.consecutive, args.gen_sample_groups_per_model, "gpt4")

    if args.gt_same_ads:
        assert args.models_same_ads, f"Need --models_same_ads to be set"
        assert args.gen_sample_groups_per_model == args.gt_sample_groups, f"Need same number of gt and model ADs to sample if --gt_same_ads is set"
        gt_samples = get_same_samples(df_llama, llama3_samples, "gt")
    else:
        gt_samples = sample_consecutive_groups(df_gpt, args.consecutive, args.gt_sample_groups, "gt")
    all_samples = llama3_samples + gpt4_samples + gt_samples
    random.shuffle(all_samples)

    annotation_samples = []
    # anno_idx,end,imdbid,start,text_gen,text_gt,name,model_name,annotate_gt
    for group in all_samples:
        group_list = []
        for df_idx, sample in group.iterrows():
            if sample['source'] == "gt":
                item = { "video": "tbd",
                        "text" : sample['text_gt'],
                        "gt_text" : sample['text_gt'],
                        "gen_text" : sample['text_gen'],
                        "gt_sample" : True,
                        "model" : sample['source'],
                        "start_s" : sample['start'],
                        "start" : str(datetime.timedelta(seconds=sample['start'])),
                        "end_s" : sample['end'],
                        "end" : str(datetime.timedelta(seconds=sample['end'])),
                        "movie" : sample['name'],
                        "movie_ad_index" : int(sample['anno_idx']),
                        "annotation_idx" : df_idx
                        }
            else:
                item = { "video": "tbd",
                        "text" : sample['text_gen'],
                        "gt_text" : sample['text_gt'],
                        "gen_text" : sample['text_gen'],
                        "gt_sample" : False,
                        "model" : sample['source'],
                        "start_s" : sample['start'],
                        "start" : str(datetime.timedelta(seconds=sample['start'])),
                        "end_s" : sample['end'],
                        "end" : str(datetime.timedelta(seconds=sample['end'])),
                        "movie" : sample['name'],
                        "movie_ad_index" : int(sample['anno_idx']),
                        "annotation_idx" : df_idx
                        }
            group_list.append(item)

        annotation_samples.append(group_list)

    with open(args.out_json, 'w') as f:
        json.dump(annotation_samples, f, indent=4, default=str)



