import json
import argparse


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", help="json annotation file to be fixed")
    parser.add_argument("missing_file", help="missing videos list file")
    parser.add_argument("output_name", help="cleaned json file name")
    return parser.parse_args()


def load_json(args):
    with open(args.json_file)as fin:
        videos = json.load(fin)
    return videos


def delete_missing(args, videos):
    with open(args.missing_file) as fread:
        missing = fread.readlines()

    for value in missing:
        try:
            videos.pop(value.strip('\n'))
        except KeyError:
            pass
    return videos


def save_json(args, videos):
    with open(args.output_name, 'w') as fout:
        fout.write(json.dumps(videos))


if __name__ == '__main__':
    args = load_args()
    videos = load_json(args)
    videos = delete_missing(args, videos)
    save_json(args, videos)
