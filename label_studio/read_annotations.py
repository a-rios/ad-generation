import sys
import os
import json
import argparse
import copy
import re
from collections import defaultdict
from typing import List, Optional
from sklearn.metrics import cohen_kappa_score # > 0.8 = good
import numpy as np
from pprint import pprint
import pandas as pd

########################################################
# 200 samples groups per model (qwen, gpt, groundtruth #
# 180 uniq -> 540                                      #
# 20 reapeated -> 60                                   #
# total: 540 + 60 x 3 (models) = 1800 per language     #
########################################################

project_ids = { 9: "German test",
                10: "German",
                11: "Italian test",
                12: "Italian"
    }

annotator_ids = { 1: "AR",
                  2: "AL",
                  3: "LF",
                  4: "VL", # == user 6
                  5: "BS",
                  6: "VL"
    }

category_dict = {   "irrelevant" : 0,
                    "missing" : 0,
                    "redundant" : 0,
                    "subjective or patronizing" : 0,
                    "wrong action" : 0,
                    "wrong object" : 0,
                    "text missing" : 0,
                    "other inaccuracy (content not in scene)" : 0,
                    "ungrammatical/not fluent" : 0,
                    "too long/complex" : 0,
                    "wrong tense" : 0,
                    "English wording/phrasing (Denglisch/Itanglese)" : 0,
                    "other AD standards violation" : 0,
                    "contextual gap" : 0,
                    "name repeated" : 0,
                    "content repeated" : 0,
                    "makes no sense/unrelated content" : 0,
                    "other incoherence" : 0,
                    "wrong character" : 0,
                    "wrong pronoun" : 0,
                    "redundant (first and last name)" : 0,
                    "missing name" : 0,
                    "bad description" : 0,
                    "misattributed action" : 0,
                    "rating" : {1 : 0,
                                2 : 0,
                                3 : 0,
                                4 : 0,
                                5 : 0},
                    'ordinal_rating': 0 # for annotator agreement calculation, needs ordinal values. only used in annotator agreement tables, per sample, no aggregate numbers here in annotations dict (stays 0)
                }

open_source_results = { "German": { "Moskau einfach" : copy.deepcopy(category_dict),
                                                    "8 Tage im August" : copy.deepcopy(category_dict),
                                                    "Baghdad in my Shadow" : copy.deepcopy(category_dict),
                                                    "Trommelwirbel" : copy.deepcopy(category_dict)
                                                 },
                                       "Italian": { "La Tentazione Di Esistere" : copy.deepcopy(category_dict),
                                                    "Riders of justice" : copy.deepcopy(category_dict),
                                                    "Stuerm" : copy.deepcopy(category_dict),
                                                    "Quanto basta" : copy.deepcopy(category_dict)
                                                  }
                                       }

gpt4_results = { "German": { "Moskau einfach" : copy.deepcopy(category_dict),
                                            "8 Tage im August" : copy.deepcopy(category_dict),
                                            "Baghdad in my Shadow" : copy.deepcopy(category_dict),
                                            "Trommelwirbel" : copy.deepcopy(category_dict)
                                                 },
                                "Italian": { "La Tentazione Di Esistere" : copy.deepcopy(category_dict),
                                             "Riders of justice" : copy.deepcopy(category_dict),
                                             "Stuerm" : copy.deepcopy(category_dict),
                                             "Quanto basta" : copy.deepcopy(category_dict)
                                                  }
                                       }
groundtruth_results = { "German": {  "Moskau einfach" : copy.deepcopy(category_dict),
                                                    "8 Tage im August" : copy.deepcopy(category_dict),
                                                    "Baghdad in my Shadow" : copy.deepcopy(category_dict),
                                                    "Trommelwirbel" : copy.deepcopy(category_dict)
                                                 },
                                       "Italian": { "La Tentazione Di Esistere" : copy.deepcopy(category_dict),
                                                    "Riders of justice" : copy.deepcopy(category_dict),
                                                    "Stuerm" : copy.deepcopy(category_dict),
                                                    "Quanto basta" : copy.deepcopy(category_dict)
                                                  }
                                       }

## intra-annotator agreement with 'repeated_groups', assuming only one annotator
intra_annotator_agreement_German = defaultdict( lambda: {   'first' : copy.deepcopy(category_dict),
                                                            'annotator_id_1' : str,
                                                            'second': copy.deepcopy(category_dict),
                                                            'annotator_id_2' : "str"
                                                         })

intra_annotator_agreement_Italian = defaultdict( lambda: {  'first' : copy.deepcopy(category_dict),
                                                            'annotator_id_1' : str,
                                                            'second': copy.deepcopy(category_dict),
                                                            'annotator_id_2' : "str"
                                                         })

def add_to_rating(model: str, lang: str, movie: str, rating: int):

    assert int(rating)>=1 and int(rating)<=5, f"Invalid rating (values 1-5): {rating}"

    if model == "qwen3":
        open_source_results[lang][movie]["rating"][rating] += 1
    elif model == "gpt4":
        gpt4_results[lang][movie]["rating"][rating] += 1
    elif model == "gt":
        groundtruth_results[lang][movie]["rating"][rating] += 1


def add_to_tags(model: str, lang: str, movie: str, tags: List[str]):

    for tag in tags:
        if tag == "Other errors (or add comment)":
            continue # TODO
        else:
            assert tag in ["irrelevant", "missing", "redundant", "subjective or patronizing", "wrong action", "wrong object", "text missing", "other inaccuracy (content not in scene)", "ungrammatical/not fluent", "too long/complex", "wrong tense", "English wording/phrasing (Denglisch/Itanglese)", "other AD standards violation", "contextual gap", "name repeated", "content repeated", "makes no sense/unrelated content", "other incoherence", "wrong character", "wrong pronoun", "redundant (first and last name)", "missing name", "bad description", "misattributed action"], f"Unknown tag: {tag}"

            if model == "qwen3":
                open_source_results[lang][movie][tag] += 1
            elif model == "gpt4":
                gpt4_results[lang][movie][tag] += 1
            elif model == "gt":
                groundtruth_results[lang][movie][tag] += 1

def get_movie_idx(data_samples: List[dict], to_name: str):
    item, i = to_name.split('_') # i: 0-2, index of sample
    return int(data_samples[int(i)]["movie_ad_index"])


def add_to_agreement_table(lang: str, model: str, movie: str, movie_ad_idx: int, annotator: str, rating: Optional[int], tags: Optional[List[str]]):
    key = f"{movie} #{movie_ad_idx}"

    if lang == "German":
        table = intra_annotator_agreement_German
    elif lang == "Italian":
        table = intra_annotator_agreement_Italian

    pass_nr = 'first'
    if key in table.keys(): # this is the second pass
        pass_nr = 'second'

    if tags is not None:
        for tag in tags:
            table[key][pass_nr][tag] +=1

    if rating is not None:
        table[key][pass_nr]['ordinal_rating'] = rating

    if pass_nr == 'first':
        table[key]['annotator_id_1'] = annotator # id 1 and 2 *should* be the same, but let's check to be sure
    else:
        table[key]['annotator_id_2'] = annotator
        assert table[key]['annotator_id_1'] == table[key]['annotator_id_2'], f"Different annotator IDs on {key}, first pass is {table[key]['annotator_id_1']}, second pass is {table[key]['annotator_id_2']}"

def calculate_cohens_kappa(lang: str):

    if lang == "German":
        table = intra_annotator_agreement_German
    else:
        table = intra_annotator_agreement_Italian
    #pprint(table, indent=2, width=100, depth=5)

    first_pass_tag_values = {"irrelevant" : [], "missing" : [],  "redundant" : [],  "subjective or patronizing" : [],
                             "wrong action" : [], "wrong object" : [], "text missing" : [],"other inaccuracy (content not in scene)" : [],
                             "ungrammatical/not fluent" : [], "too long/complex" : [], "wrong tense" : [],
                             "English wording/phrasing (Denglisch/Itanglese)" : [], "other AD standards violation" : [],
                             "contextual gap" : [], "name repeated" : [], "content repeated" : [], "makes no sense/unrelated content" : [],
                             "other incoherence" : [], "wrong character" : [], "wrong pronoun" : [], "redundant (first and last name)" : [],
                             "missing name" : [], "bad description" : [], "misattributed action" : []}
    first_pass_rating_values = [] # list with length = number of samples in repeated_groups (60)
    second_pass_tag_values = {"irrelevant" : [], "missing" : [],  "redundant" : [],  "subjective or patronizing" : [],
                             "wrong action" : [], "wrong object" : [], "text missing" : [],"other inaccuracy (content not in scene)" : [],
                             "ungrammatical/not fluent" : [], "too long/complex" : [], "wrong tense" : [],
                             "English wording/phrasing (Denglisch/Itanglese)" : [], "other AD standards violation" : [],
                             "contextual gap" : [], "name repeated" : [], "content repeated" : [], "makes no sense/unrelated content" : [],
                             "other incoherence" : [], "wrong character" : [], "wrong pronoun" : [], "redundant (first and last name)" : [],
                             "missing name" : [], "bad description" : [], "misattributed action" : []}  # list values: length = number of samples in repeated_groups (60)
    second_pass_rating_values = []  # list with length = number of samples in repeated_groups (60)
    tags_cohens_kappa = {"irrelevant" : 0, "missing" : 0,  "redundant" : 0,  "subjective or patronizing" : 0,
                             "wrong action" : 0, "wrong object" : 0, "text missing" : 0,"other inaccuracy (content not in scene)" : 0,
                             "ungrammatical/not fluent" : 0, "too long/complex" : 0, "wrong tense" : 0,
                             "English wording/phrasing (Denglisch/Itanglese)" : 0, "other AD standards violation" : 0,
                             "contextual gap" : 0, "name repeated" : 0, "content repeated" : 0, "makes no sense/unrelated content" : 0,
                             "other incoherence" : 0, "wrong character" : 0, "wrong pronoun" : 0, "redundant (first and last name)" : 0,
                             "missing name" : 0, "bad description" : 0, "misattributed action" : 0}


    for idx in table:
        # TODO assert that annotator ids match
        first_pass_rating_values.append(table[idx]['first']['ordinal_rating'])
        second_pass_rating_values.append(table[idx]['second']['ordinal_rating'])

        for key in table[idx]['first'].keys():
            if key not in ['rating', 'ordinal_rating']:
                first_pass_tag_values[key].append(table[idx]['first'][key])
        for key in table[idx]['second'].keys():
            if key not in ['rating', 'ordinal_rating']:
                second_pass_tag_values[key].append(table[idx]['second'][key])
    #pprint(first_pass_tag_values, indent=2)
    #pprint(second_pass_tag_values, indent=2)
    for (tag1, valuelist1), (tag2, valuelist2) in zip(first_pass_tag_values.items(), second_pass_tag_values.items()):
        assert tag1 == tag2, f"Cannot calculate Cohen's kappa for different tags: first pass tag={tag1}, second pass tag={tag2}"
        kappa = cohen_kappa_score(valuelist1, valuelist2)
        if np.isnan(kappa): # if there is no variance, i.e. everything is 0 or 1 -> kappa will be NaN (division by 0), but this is perfect agreement
            kappa = 1.0
        tags_cohens_kappa[tag1] = kappa

    ratings_cohens_kappa = cohen_kappa_score(first_pass_rating_values, second_pass_rating_values, weights='quadratic') # quadratic: penalty increases with the square of the distance, i.e., a rating of 1 vs. 2 is better than 1 vs. 5
    return tags_cohens_kappa, ratings_cohens_kappa

def get_stats_kappa(tags: dict):
    # return mean, media, minimum with tag, maximum with tag, std dev
    kappas = [k for k in tags_cohens_kappa.values()]
    min_item = min(tags_cohens_kappa.items(), key=lambda x: x[1])
    max_item = max(tags_cohens_kappa.items(), key=lambda x: x[1])

    out_string = f"""
                mean: {np.mean(kappas)}
                median: {np.median(kappas)}
                lowest: '{min_item[0]}' = {min_item[1]:.3f}
                highest: '{max_item[0]}' = {max_item[1]:.3f}
                std dev: {np.std(kappas, ddof=1):.3f}
        """
    return out_string

def get_error_counts(annotations: defaultdict, lang: str):
    rows = []
    for movie_name, counts in annotations[lang].items():
            row = {'movie': movie_name}
            row['total error tags'] = 0
            row['content errors'] = 0
            row['grammar errors'] = 0
            row['coherence errors'] = 0
            row['character errors'] = 0

            # Add all error counts
            total_error_tags = 0
            for label, count in counts.items():
                if label not in ['rating', 'ordinal_rating']:
                    row[label] = count
                    row['total error tags'] += count

                    # content errors
                    if label in ["irrelevant", "missing", "redundant", "subjective or patronizing", "wrong action", "wrong object", "text missing", "other inaccuracy (content not in scene)"]:
                        row['content errors'] += count
                    if label in ["ungrammatical/not fluent", "too long/complex", "wrong tense", "English wording/phrasing (Denglisch/Itanglese)", "other AD standards violation"]:
                        row['grammar errors'] += count
                    if label in ["contextual gap", "name repeated", "content repeated", "makes no sense/unrelated content", "other incoherence"]:
                        row['coherence errors'] += count
                    if label in ["wrong character", "wrong pronoun", "redundant (first and last name)", "missing name", "bad description", "misattributed action"]:
                        row['character errors'] += count

                elif label == "rating":
                    row['rating_1'] = counts['rating'][1]
                    row['rating_2'] = counts['rating'][2]
                    row['rating_3'] = counts['rating'][3]
                    row['rating_4'] = counts['rating'][4]
                    row['rating_5'] = counts['rating'][5]

            ratings = counts['rating']
            total_rated = sum(counts['rating'].values())
            if total_rated > 0:
                mean_quality = sum(r * c for r, c in ratings.items()) / total_rated
            else:
                mean_quality = 0

            row['mean_quality'] = mean_quality
            row['severe_errors'] = ratings[1] + ratings[2]
            rows.append(row)
    df = pd.DataFrame(rows)
    totals = df.sum(numeric_only=True)
    totals['movie'] = 'total'
    total_rated = totals['rating_1'] + totals['rating_2'] + totals['rating_3'] + totals['rating_4'] + totals['rating_5']
    if total_rated > 0:
        totals['mean_quality'] = (
            1 * totals['rating_1'] +
            2 * totals['rating_2'] +
            3 * totals['rating_3'] +
            4 * totals['rating_4'] +
            5 * totals['rating_5']
        ) / total_rated
    else:
        totals['mean_quality'] = 0
    df_with_totals = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)

    return df_with_totals

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ls_json', type=str, help='label studio json export file')
    parser.add_argument('--out_dir', type=str, required=True, help='write output to this directory. Will write 3 files (one for each model: ground truth, gpt4, open source)')
    parser.add_argument('--print_cohens_kappa_per_label', action='store_true', help='print kappa for each label')
    parser.add_argument('--print_cohens_kappa_per_group', action='store_true', help='print kappa for groups of labels (content, grammar, coherence, characters)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()


    with open(args.ls_json, 'r') as f:
        data = json.load(f)

        # json is all one language, just check first movie name
        lang = ""
        if data[0]['data']['movie'] in ["Moskau einfach", "8 Tage im August", "Baghdad in my Shadow", "Trommelwirbel"]:
            lang = "German"
        elif data[0]['data']['movie']  in ["La Tentazione Di Esistere", "Riders of justice", "Stuerm", "Quanto basta"]:
            lang = "Italian"

        for sample in data:
            annotations = sample["annotations"]
            if len(annotations) > 0:
                data = sample["data"]
                movie = data["movie"]
                assert movie in ["Moskau einfach", "8 Tage im August", "Baghdad in my Shadow", "Trommelwirbel", "La Tentazione Di Esistere", "Riders of justice", "Stuerm", "Quanto basta" ], f"Unknown movie: {movie}"

                model = data["ad_block"][0]["model"] # all 3 ADs in one group are from the same movie, we can just take the value of the first item
                assert model in ["qwen3", "gpt4", "gt"], f"Unknown model name (qwen3, gpt4, gt): {modelname}"

                repeated_group = data["ad_block"][0]["repeated_group"] # same here
                data_samples = [i for i in data["ad_block"]]

                # movie_idxs = [data_dict["movie_ad_index"] for data_dict in  data["ad_block"]]

                for annotation in annotations:
                    annotator = annotator_ids[int(annotation["completed_by"])]
                    for result_dict in annotation["result"]: # result = list of dicts with annotations
                        if result_dict['origin'] == "manual": # ignore 'predictions' (timespans used for easier video navigation)
                            # print(result_dict)
                            if result_dict["type"] == "rating":
                                stars = result_dict["value"]["rating"]
                                add_to_rating(model, lang, movie, stars)
                                if repeated_group:
                                    movie_ad_idx = get_movie_idx(data_samples, result_dict["to_name"])
                                    add_to_agreement_table(lang, model, movie, movie_ad_idx, annotator, stars, None)
                            elif result_dict["type"] == "choices":
                                tags = result_dict["value"]["choices"]
                                add_to_tags(model, lang, movie, tags)
                                if repeated_group:
                                    add_to_agreement_table(lang, model, movie, movie_ad_idx, annotator, None, tags)


        # print(json.dumps(groundtruth_results, indent=2))
        # print()
        # print(json.dumps(open_source_results, indent=2))
        # print()
        # print(json.dumps(gpt4_results, indent=2))
        # print()

        #######################
        #  annotations stats  #
        #######################

        OS_df_errors = get_error_counts(open_source_results, lang)
        GT_df_errors = get_error_counts(groundtruth_results, lang)
        GPT_df_errors = get_error_counts(gpt4_results, lang)

        OS_outname = os.path.join(args.out_dir, f"{lang}_open_source.annotations.csv")
        GT_outname = os.path.join(args.out_dir, f"{lang}_groundtruth.annotations.csv")
        GPT_outname = os.path.join(args.out_dir, f"{lang}_gpt4.annotations.csv")


        OS_df_errors.to_csv(OS_outname, encoding='utf-8', index=False)
        GT_df_errors.to_csv(GT_outname, encoding='utf-8', index=False)
        GPT_df_errors.to_csv(GPT_outname, encoding='utf-8', index=False)


        #############################
        # intra-annotator agreement #
        #############################
        tags_cohens_kappa, ratings_cohens_kappa = calculate_cohens_kappa(lang)
        print("************************************************************************")
        print(f"Intra-annotator agreement (Cohen's kappa):")
        print(f"Ratings Cohen's kappa: {ratings_cohens_kappa}")
        print(f"Tags Cohen's kappa, overall:")
        print(get_stats_kappa(tags_cohens_kappa))

        if args.print_cohens_kappa_per_label:
            for tag, kappa in tags_cohens_kappa.items():
                print(f"{tag}: {kappa}")

        if args.print_cohens_kappa_per_group:
            content_kappas = {k:v for k,v in tags_cohens_kappa.items() if k in ["irrelevant", "missing", "redundant", "subjective or patronizing", "wrong action", "wrong object", "text missing", "other inaccuracy (content not in scene)"]}
            print(f"\tContent tags:")
            print(get_stats_kappa(content_kappas))

            grammar_kappas = {k:v for k,v in tags_cohens_kappa.items() if k in ["ungrammatical/not fluent", "too long/complex", "wrong tense", "English wording/phrasing (Denglisch/Itanglese)", "other AD standards violation"]}
            print(f"\tGrammar tags:")
            print(get_stats_kappa(grammar_kappas))


            coherence_kappas = {k:v for k,v in tags_cohens_kappa.items() if k in ["contextual gap", "name repeated", "content repeated", "makes no sense/unrelated content", "other incoherence"]}
            print(f"\tCoherence tags:")
            print(get_stats_kappa(coherence_kappas))

            character_kappas = {k:v for k,v in tags_cohens_kappa.items() if k in ["wrong character", "wrong pronoun", "redundant (first and last name)", "missing name", "bad description", "misattributed action"]}
            print(f"\tCharacter tags:")
            print(get_stats_kappa(character_kappas))
        print("************************************************************************")


