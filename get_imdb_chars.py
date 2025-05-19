from imdb import Cinemagoer
import json
import argparse
import requests
import os



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', metavar='PATH', type=str, help="json to write output")
    parser.add_argument('--movie_ids', type=str, nargs="+", help="list of imdb movie ids (tt...)")
    parser.add_argument('--image_dir', metavar='PATH', type=str, help="output dir to write images")
    args = parser.parse_args()

    ia = Cinemagoer()
    imdb_ids = args.movie_ids
    os.makedirs(args.image_dir, exist_ok=True)

    result = {}

    for imdb_id in imdb_ids:
        print(imdb_id)
        movie = ia.get_movie(imdb_id[2:])
        cast_info = []
        for person in movie.get('cast', []):

            # download picture, if available
            ia.update(person, info=['main'])

            filename = None

            try:
                photo_url = person.data['headshot']

                if photo_url:
                    response = requests.get(photo_url)
                    if response.status_code == 200:
                        filename = f"{person.personID}.jpg"
                        with open(os.path.join(args.image_dir, filename), 'wb') as f:
                            f.write(response.content)

            except KeyError:
                print(f"No headshot found for {person['name']}")

            person = {
                "id": person.personID,
                "role": person.currentRole if person.currentRole else "",
                "name": person['name'],
                "img" : os.path.join(args.image_dir, filename) if filename else None
                }

            cast_info.append(person)

        result[imdb_id] = cast_info


    with open(args.json, 'w') as f:
        json.dump(result, f, indent=4, default=str)


