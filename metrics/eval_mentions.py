import os
import sys
import ast
import re
import json
import argparse
import pandas as pd
from typing import List, Dict, Tuple



def insert_offsets_in_order(preds: List[Dict], text: str):
    """
    Assign (start,end) to each prediction by finding the next occurrence of its span
    at/after a moving cursor. If not found, mark start/end as None/None.
    """
    cursor = 0
    for p in preds:
        span = p["text"]
        pattern = r'\b' + re.escape(span) + r'\b'
        m = re.search(pattern, text[cursor:])
        if not m:
            p["start"] = None
            p["end"] = None
        else:
            start = cursor + m.start()
            end = start + len(span)
            p["start"] = start
            p["end"] = end
            cursor = end
    return preds


def score_sample(preds: List[Dict], gold: List[Dict], idx: int) -> Tuple[Tuple[int,int,int], Tuple[int,int,int]]:
    """
    Returns:
      (tp_strict, fp_strict, fn_strict), (tp_span, fp_span, fn_span)

    - Strict: (start,end,label) must match one of the gold labels for that span (1-to-1).
    - Span-only: (start,end) must match (1-to-1).
    - Predictions with start/end == None are always FP.
    - Duplicates are handled by 1-to-1 matching (extras become FP).
    """
    # Expand gold into strict units: one item per (start,end,label)
    gold_strict = []
    for g in gold:
        s, e = g["start"], g["end"]
        gold_strict.append((s, e, g['labels'][0])) # there is never more than one label for a span in gold
    used_strict = [False] * len(gold_strict)

    # Also keep span-only list (one item per span, regardless of labels)
    gold_spans = [(g["start"], g["end"]) for g in gold]
    used_span = [False] * len(gold_spans)

    tp_strict = fp_strict = fn_strict = 0
    tp_span = fp_span = fn_span = 0

    # Process predictions in given order (None offsets will be handled as FP)
    for p in preds:
        ps, pe, pl = p.get("start"), p.get("end"), p.get("label")

        # Strict matching
        if ps is None or pe is None:
            fp_strict += 1
        else:
            matched = False
            for i, (gs, ge, gl) in enumerate(gold_strict):
                if not used_strict[i] and ps == gs and pe == ge and pl == gl:
                    used_strict[i] = True
                    tp_strict += 1
                    matched = True
                    break
            if not matched:
                fp_strict += 1

        # Span-only matching
        if ps is None or pe is None:
            fp_span += 1
        else:
            matched = False
            for j, (gs, ge) in enumerate(gold_spans):
                if not used_span[j] and ps == gs and pe == ge:
                    used_span[j] = True
                    tp_span += 1
                    matched = True
                    break
            if not matched:
                fp_span += 1
                print(f"{idx} FP: {p}, gold: {gold}")

    # Any unused gold are FNs
    fn_strict = used_strict.count(False)
    fn_span = used_span.count(False)

    return (tp_strict, fp_strict, fn_strict), (tp_span, fp_span, fn_span)


def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1




def main(args):
    gold_df = pd.read_csv(args.gold_mentions)

    with open(args.predicted_mentions, 'r') as f:
        predicted_mentions = f.readlines()

    TP_strict=0
    FP_strict=0
    FN_strict=0
    TP_span=0
    FP_span=0
    FN_span=0

    assert len(gold_df) == len(predicted_mentions), f"Gold and predicted mentions have a different number of samples: gold {len(gold_df)} vs. predicted {len(predicted_mentions)}"

    for (idx, gold_row), predicted in zip(gold_df.iterrows(), predicted_mentions):
        full_text = gold_row['text']

        if pd.isna(gold_row['label']):
            gold_json = []
        else:
            gold_json = json.loads(gold_row['label'])
        predicted_json = json.loads(predicted)

        ## if both empty: skip (no error, but no TP/TN either, since there was nothing to predict)
        if len(gold_json) == len(predicted_json) and len(gold_json) == 0:
            continue

        elif len(gold_json) == 0 and len(predicted_json) >0: # all FP
            FP_strict += len(predicted_json)
            FP_span += len(predicted_json)
            # print(f"{idx} FP: {predicted_json}, gold: {gold_json}")
            continue

        elif len(predicted_json) ==0 and len(gold_json) >0: # all FN
            FN_strict += len(predicted_json)
            FN_span += len(predicted_json)
            # print(f"{idx} FN: {predicted_json}, gold: {gold_json}")
            continue

        else:
            # get start/end times for predicted spans, note: predictions can have overlapping spans (not allowed), repeat labels (e.g. too many of the same pronoun) (FP) or not enough (FN)
            # -> we need to keep track of what's already been matched, i.e. the last matched index since labels go left to right
            predicted_json = insert_offsets_in_order(predicted_json, full_text)
            (tp_strict, fp_strict, fn_strict), (tp_span, fp_span, fn_span) = score_sample(predicted_json, gold_json, idx)
            TP_strict += tp_strict
            TP_span += tp_span
            FP_strict += fp_strict
            FP_span += fp_span
            FN_strict += fn_strict
            FN_span += fn_span


            # print(full_text)
            # print("predicted:")
            # print(predicted_json)
            # print("gold:")
            # print(gold_json)
            # print("-----------")


    precision_strict, recall_strict, f1_strict = precision_recall_f1(TP_strict, FP_strict, FN_strict)
    precision_span, recall_span, f1_span = precision_recall_f1(TP_span, FP_span, FN_span)

    print(f"""
Span evaluation (only spans match, labels not checked):
    precision: {precision_span:,.3f}, recall: {recall_span:,.3f}, F1: {f1_span:,.3f}

Strict evaluation (spans and labels must match):
    precision: {precision_strict:,.3f}, recall: {recall_strict:,.3f}, F1: {f1_strict:,.3f}
""")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_mentions', default=None, type=str, help='csv with gold mentions')
    parser.add_argument('--predicted_mentions', default=None, type=str, help='file with predicted mentions (one list of dicts per line)')

    args = parser.parse_args()

    main(args)