import json
import pandas as pd
import argparse
from pathlib import Path
import os
import datetime
import ffmpeg



def get_paths(name, start, end, video_input_dir, video_output_dir, video_prefix_json):
    in_name = os.path.join(video_input_dir, f"{name}.mp4")
    out_name = os.path.join(video_output_dir, f"{name}-{start}-{end}.mp4")
    if video_prefix_json is not None:
        link_name = os.path.join(video_prefix_json, f"{name}-{start}-{end}.mp4")
    else:
        link_name = out_name

    return in_name, out_name, link_name


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
                for sample in group:
                    start = sample['start_s']
                    end = sample['end_s']
                    duration = end - start

                    # name = sample['movie']
                    # in_name = os.path.join(args.video_input_dir, f"{name}.mp4")
                    # out_name = os.path.join(args.video_output_dir, f"{name}-{sample['start']}-{sample['end']}.mp4")
                    # if args.video_prefix_json is not None:
                    #     link_name = os.path.join(args.video_prefix_json, f"{name}-{sample['start']}-{sample['end']}.mp4")
                    # else:
                    #     link_name = out_name

                    in_name, out_name, link_name = get_paths(sample['movie'], sample['start'], sample['end'], args.video_input_dir, args.video_output_dir, args.video_prefix_json)

                    if not os.path.exists(out_name): # don't overwrite clips if already extracted
                        (
                            ffmpeg
                            .input(in_name, ss=start)
                            .output(out_name, t=duration, vcodec='libx264', acodec='aac')
                            .run(overwrite_output=True)
                        )

                    sample['video'] = link_name
        else:
            new_data = []
            for group in data:
                start = group[0]['start_s']
                end = group[-1]['end_s']
                duration = end - start

                in_name, out_name, link_name = get_paths(group[0]['movie'], group[0]['start'], group[-1]['end'], args.video_input_dir, args.video_output_dir, args.video_prefix_json)

                if not os.path.exists(out_name): # don't overwrite clips if already extracted
                    (
                        ffmpeg
                        .input(in_name, ss=start)
                        .output(out_name, t=duration, vcodec='libx264', acodec='aac')
                        .run(overwrite_output=True)
                    )

                group_dict = {"video" : link_name,
                              "samples" : group}
                new_data.append(group_dict)
            data = new_data



        with open(args.out_json, 'w') as f:
            json.dump(data, f, indent=4, default=str)
