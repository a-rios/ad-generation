import json
import pandas as pd
import argparse
from pathlib import Path
import os
import datetime
import ffmpeg


def create_predictions(group, fps):
    predictions = {'model_version': 'dummy',
                   'score': 0}
    result = []

    for i,sample in enumerate(group['ad_block'], start=1):
        value = {
            "ranges": [
                {
                "start": sample['first_frame']+1, # frames in label studio interface start at 1, not 0
                "end": sample['last_frame']+1
                }
            ],
            "timelinelabels": [ f"AD {i}" ]
            }
        i_dict = {'value': value,
                  'id': sample['region_id'],
                  'from_name': 'ad_segment_labels',
                  'to_name' : 'video_player',
                  'type': 'timelinelabels',
                  'readonly' : True,
                  'locked' : True}
        result.append(i_dict)

    predictions['result'] = result
    return [predictions]

def get_original_fps(video_path):
    probe = ffmpeg.probe(video_path)

    video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
    if not video_streams:
        raise ValueError("No video stream found")

    r_frame_rate = video_streams[0]['r_frame_rate']
    num, denom = map(int, r_frame_rate.split('/'))
    fps = num / denom
    return fps

def get_paths(name, start, end, video_input_dir, video_output_dir, video_prefix_json):
    in_name = os.path.join(video_input_dir, f"{name}.mp4")
    out_name = os.path.join(video_output_dir, f"{name}-{start}-{end}.mp4")
    if video_prefix_json is not None:
        link_name = os.path.join(video_prefix_json, f"{name}-{start}-{end}.mp4")
    else:
        link_name = out_name

    orig_fps = get_original_fps(in_name)

    return in_name, out_name, link_name, orig_fps


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', metavar='PATH', type=str,  help="json with samples ADs for annotation")
    parser.add_argument('--out_json', metavar='PATH', type=str,  help="output json with path to video clips (set prefix if not an absolute path)")
    parser.add_argument('--video_input_dir', metavar='PATH', type=str, help="path to input videos")
    parser.add_argument('--video_output_dir', metavar='PATH', type=str, help="path to save extracted clips")
    parser.add_argument('--video_prefix_json', type=str, help="prefix for json to where clips will be served (e.g. localhost:port")
    parser.add_argument('--individual_clips', action="store_true", help="extract and link clips for each sample individually (default: extract one clip per group of samples (from start of first to end of last))")
    args = parser.parse_args()

    with open(args.json, 'r') as f:
        data = json.load(f)


        if args.individual_clips:
            for group in data:
                for sample in group['ad_block']:
                    start = sample['abs_start_s']
                    end = sample['abs_end_s']
                    duration = end - start

                    in_name, out_name, link_name, fps = get_paths(sample['movie'], sample['abs_start'], sample['abs_end'], args.video_input_dir, args.video_output_dir, args.video_prefix_json)

                    if not os.path.exists(out_name): # don't overwrite clips if already extracted
                        (
                            ffmpeg
                            .input(in_name, ss=start)
                            .output(out_name, t=duration, vcodec='libx264', acodec='aac')
                            .run(overwrite_output=True)
                        )

                    sample['video'] = link_name
                    # TODO add first and last frame to sample
        else:
            new_data = []
            for group in data:
                start = group['ad_block'][0]['abs_start_s']
                end = group['ad_block'][-1]['abs_end_s']
                duration = end - start

                in_name, out_name, link_name, fps = get_paths(group['ad_block'][0]['movie'], group['ad_block'][0]['abs_start'], group['ad_block'][-1]['abs_end'], args.video_input_dir, args.video_output_dir, args.video_prefix_json)

                for sample in group['ad_block']:
                    sample['first_frame'] = int(sample['rel_start_s'] * fps)
                    sample['last_frame'] = int(sample['rel_end_s'] * fps)
                    sample['frames'] = f"{sample['first_frame']+1}-{sample['last_frame']+1}" # frames in label studio interface start at 1, not 0

                print(f"Extracting {out_name}")
                if not os.path.exists(out_name): # don't overwrite clips if already extracted
                    (
                        ffmpeg
                        .input(in_name, ss=start)
                        .output(out_name, t=duration, vcodec='libx264', acodec='aac')
                        .run(overwrite_output=True)
                    )

                predictions = create_predictions(group, fps)

                group_dict = { "data": {
                                    "video" : link_name,
                                    "fps" : int(fps),
                                    "ad_block" : group["ad_block"]
                                    },
                              "predictions" : predictions}
                new_data.append(group_dict)
            data = new_data



        with open(args.out_json, 'w') as f:
            json.dump(data, f, indent=4, default=str)

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
