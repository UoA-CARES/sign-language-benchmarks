import json
import argparse
import os

DATA_PATH = "data/wlasl/wlasl-uncompressed"

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file', help='json annotation file to be fixed')
    parser.add_argument('missing_file', help='missing videos list file')
    parser.add_argument('output_name', help='cleaned json file name')
    parser.add_argument('directory', help='path to the uncompressed archive')
    return parser.parse_args()


def load_json(args):
    with open(args.json_file)as fin:
        videos = json.load(fin)
    return videos


def delete_missing(args, videos):
    # Read the missing text file
    with open(args.missing_file) as fread:
        missing = fread.readlines()

    # Delete the video from json if in missing file
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
    os.chdir(args.directory)
    videos = load_json(args)
    videos = delete_missing(args, videos)
    save_json(args, videos)
