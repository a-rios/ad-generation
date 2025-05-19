from imdb import Cinemagoer
import json
import argparse
import requests
import os



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imdb_json', metavar='PATH', type=str, help="json with imdb data")
    parser.add_argument('--image_dir', type=str, help="directory with headshots (Firstname_Lastname.jpg)")
    parser.add_argument('--out_json', type=str, help="output json")
    args = parser.parse_args()

    with open(args.imdb_json, 'r') as f:
        data = json.load(f)

        for movie in data:
            for actor in data[movie]:
                if not actor['img']:
                    img_name = f"{actor['name'].replace(' ', '_')}.jpg"
                    img_path = os.path.join(args.image_dir, img_name)
                    if os.path.isfile(img_path):
                        actor['img'] = img_path




    with open(args.out_json, 'w') as f:
         json.dump(data, f, indent=4, default=str)


