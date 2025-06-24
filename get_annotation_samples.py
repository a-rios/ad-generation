import json
import pandas as pd
import argparse
from pathlib import Path
import os
import random
import datetime
from collections import Counter

random.seed(42)


def shuffle_no_neighbors(numbers: list[int], max_attempts: int = 10):
    """
    Shuffles a list of integers such that no identical integers are neighbors.
    Some integers are repeated twice, others are unique.
    Returns a shuffled list, or raises an error if a solution cannot be found
    within max_attempts.
    """
    initial_len = len(numbers)

    for attempt in range(max_attempts):
        current_counts = Counter(numbers)
        shuffled_list = []
        last_picked_value = None

        # Loop until the shuffled list is full or we get stuck
        while len(shuffled_list) < initial_len:
            # Find available choices that are not the last picked value
            available_choices = []
            for num, count in current_counts.items():
                if count > 0 and num != last_picked_value:
                    available_choices.append(num)

            if not available_choices:
                break

            next_num = random.choice(available_choices)

            shuffled_list.append(next_num)
            current_counts[next_num] -= 1
            last_picked_value = next_num

        # If we successfully built the whole list, return it
        if len(shuffled_list) == initial_len:
            return shuffled_list

    raise ValueError(f"Could not find a valid shuffle after {max_attempts} attempts.")


def sample_consecutive_starts(df, group_size, n_groups, n_repeated_groups):

    assert n_repeated_groups < n_groups, f"Cannot repeat {n_repeated_groups} from sample size {n_groups}"

    total_rows = len(df)
    max_full_groups = total_rows // group_size

    # non-overlapping group start indices
    group_starts = [i * group_size for i in range(max_full_groups)]

    n_to_sample = n_groups - n_repeated_groups

    # sample group start indices
    sampled_starts = random.sample(group_starts, min(n_to_sample, len(group_starts)))

    return sampled_starts

def get_rows(df, group_starts, group_size, model):
    groups = []
    for start in group_starts:
        group = df.iloc[start:start + group_size].copy()
        group["source"] = model
        group['group_start'] = start
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
    parser.add_argument('--repeated_groups', type=int, help="repeat N group samples from each model and gt (to get intra-annotator agreement) ")
    parser.add_argument('--out_json', metavar='PATH',type=str, help="output json (format for label studio")
    args = parser.parse_args()


    if args.gt_same_ads:
        assert args.models_same_ads, f"Need --models_same_ads to be set"
        assert args.gen_sample_groups_per_model == args.gt_sample_groups, f"Need same number of gt and model ADs to sample if --gt_same_ads is set"


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
        sampled_starts = sample_consecutive_starts(df_llama, args.consecutive, args.gen_sample_groups_per_model, args.repeated_groups)
        repeated_llama3_starts = random.sample(sampled_starts, args.repeated_groups)
        llama3_starts = sampled_starts + repeated_llama3_starts
        repeated_gpt4_starts = random.sample(sampled_starts, args.repeated_groups)
        gpt4_starts = sampled_starts + repeated_gpt4_starts

        if args.gt_same_ads:
            repeated_gt_starts = random.sample(sampled_starts, args.repeated_groups)
            gt_starts = sampled_starts + repeated_gt_starts

    else:
        llama3_starts = sample_consecutive_starts(df_llama, args.consecutive, args.gen_sample_groups_per_model, args.repeated_groups)
        gpt4_starts = sample_consecutive_starts(df_gpt, args.consecutive, args.gen_sample_groups_per_model, args.repeated_groups)
        gt_starts = sample_consecutive_starts(df_gpt, args.consecutive, args.gen_sample_groups_per_model, args.repeated_groups) # can sample gt from any of the dfs

        repeated_llama3_starts = random.sample(llama3_starts, args.repeated_groups)
        llama3_starts = llama3_starts + repeated_llama3_starts
        repeated_gpt4_starts = random.sample(gpt4_starts, args.repeated_groups)
        gpt4_starts = gpt4_starts + repeated_gpt4_starts
        repeated_gt_starts = random.sample(gt_starts, args.repeated_groups)
        gt_starts = gt_starts + repeated_gt_starts

    llama3_starts = shuffle_no_neighbors(llama3_starts)
    gpt4_starts = shuffle_no_neighbors(gpt4_starts)
    gt_starts = shuffle_no_neighbors(gt_starts)

    # get rows from dfs
    llama3_samples = get_rows(df_llama, llama3_starts, args.consecutive, "llama3")
    gpt4_samples = get_rows(df_gpt, gpt4_starts, args.consecutive, "gpt4")
    gt_samples = get_rows(df_gpt, gt_starts, args.consecutive, "gt")
    all_samples = llama3_samples + gpt4_samples + gt_samples
    random.shuffle(all_samples)

    annotation_samples = []
    # anno_idx,end,imdbid,start,text_gen,text_gt,name,model_name,annotate_gt
    for group in all_samples:
        group_list = []

        # need start times in the clip, not the full video -> get start of first AD in group (start of clip) = offset
        offset = group.iloc[0]['start']

        for i, (_, sample) in enumerate(group.iterrows(), start=1):

            start = str(datetime.timedelta(seconds=sample['start']))
            end = str(datetime.timedelta(seconds=sample['end']))

            if sample['source'] == "gt":

                item = {
                        "text" : sample['text_gt'],
                        "gt_text" : sample['text_gt'],
                        "gen_text" : sample['text_gen'],
                        "gt_sample" : True,
                        "model" : sample['source'],
                        "abs_start_s" : sample['start'],
                        "abs_start" : start,
                        "abs_end_s" : sample['end'],
                        "abs_end" : end,
                        "rel_start_s": sample['start']-offset,
                        "rel_end_s": sample['end']-offset,
                        "first_frame" : "tbd",
                        "last_frame" : "tbd", # insert when video clip is linked
                        "movie" : sample['name'],
                        "movie_ad_index" : int(sample['anno_idx']),
                        "annotation_idx" : i,
                        "region_id" : f"region_{i}",
                        "id" : f"id{i}",
                        "repeated_group" : sample['group_start'] in repeated_gt_starts
                        }
            else:
                if sample['source'] == "llama3":
                    repeated = sample['group_start'] in repeated_llama3_starts
                if sample['source'] == "gpt4":
                    repeated = sample['group_start'] in repeated_gpt4_starts
                item = {
                        "text" : sample['text_gen'],
                        "gt_text" : sample['text_gt'],
                        "gen_text" : sample['text_gen'],
                        "gt_sample" : False,
                        "model" : sample['source'],
                        "abs_start_s" : sample['start'],
                        "abs_start" : start,
                        "abs_end_s" : sample['end'],
                        "abs_end" : end,
                        "rel_start_s": sample['start']-offset,
                        "rel_end_s": sample['end']-offset,
                        "first_frame" : "tbd",
                        "last_frame" : "tbd", # insert when video clip is linked
                        "movie" : sample['name'],
                        "movie_ad_index" : int(sample['anno_idx']),
                        "annotation_idx" : i,
                        "region_id" : f"region_{i}",
                        "id" : f"id{i}",
                        "repeated_group" : repeated
                        }
            group_list.append(item)
            group_dict = {'video': "tbd",
                          'ad_block': group_list,
                          "predictions": []}

        annotation_samples.append(group_dict)


    with open(args.out_json, 'w') as f:
        json.dump(annotation_samples, f, indent=4, default=str)


# [
#   {
#     "data": {
#       "video": "http://localhost:8081/Beautiful_Minds-0:05:31.360000-0:05:36.840000.mp4",
#       "ad_block": [
#         {"id": "id1", "title": "Title One", "text": "Body for one", "start_time": "2s", "end_time": "10s", "region_id": "ad_region_0"},
#         {"id": "id2", "title": "Title Two", "text": "Body for two", "start_time": "15s", "end_time": "25s", "region_id": "ad_region_1"},
#         {"id": "id3", "title": "Title Three", "text": "Body for three", "start_time": "25s", "end_time": "45s", "region_id": "ad_region_2"}
#       ]
#     },
#     "predictions": [
#       {
#         "model_version": "ad_segments_from_json_v1",
#         "score": 0.99,
#         "result": [
#           {
#             "value": {
#             "ranges": [
#               {
#                 "start": 2,
#                 "end": 10
#               }
#             ],
#             "timelinelabels": [
#               "AD 1"
#             ]
#           },
#             "id": "ad_region_0",
#             "from_name": "ad_segment_labels",
#             "to_name": "video_player",
#             "type": "timelinelabels",
#             "readonly": true
#           },
#           {
#             "value": {
#             "ranges": [
#               {
#                 "start": 15,
#                 "end": 25
#               }
#             ],
#             "timelinelabels": [
#               "AD 2"
#               ]
#             },
#             "id": "ad_region_1",
#             "from_name": "ad_segment_labels",
#             "to_name": "video_player",
#             "type": "timelinelabels",
#             "readonly": true
#           },
#           {
#             "value": {
#             "ranges": [
#               {
#                 "start": 25,
#                 "end": 45
#               }
#             ],
#             "timelinelabels": [
#               "AD 3"
#             ]
#           },
#             "id": "ad_region_2",
#             "from_name": "ad_segment_labels",
#             "to_name": "video_player",
#             "type": "timelinelabels",
#             "readonly": true
#           }
#         ]
#       }
#     ]
#   }
# ]

