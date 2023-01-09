import json
import os
import argparse

SUBSETS = ['train', 'test', 'val']


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file', help='json annotation file to be used')
    parser.add_argument('directory', help='path to folder rawframes')
    return parser.parse_args()


def delete_existing_annotations():
    # Delete existing annotation files
    for subset in SUBSETS:
        try:
            os.remove(f'{subset}_annotations.txt')
        except:
            pass


def load_annotations(args):
    # Load the json labels
    with open(args.json_file) as fin:
        videos = json.load(fin)
    return videos


def write_annotations(videos):
    # Create the annotation files
    for video_id in videos:
        class_id = videos[video_id]['action'][0]
        subset = videos[video_id]['subset']
        directory = f'rawframes/{subset}/{video_id}'

        frames = len([frame for frame in os.listdir(directory)
                     if os.path.isfile(os.path.join(directory, frame))])
        with open(f'{subset}_annotations.txt', 'a') as fout:
            fout.write(f'{subset}/{video_id} {frames} {class_id}\n')


if __name__ == '__main__':
    args = load_args()
    videos = load_annotations(args)
    os.chdir(args.directory)
    delete_existing_annotations()
    write_annotations(videos)
