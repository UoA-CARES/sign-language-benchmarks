import json
import os
import argparse

SUBSETS = ['train', 'test', 'val']

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file', help='json annotation file to be fixed')
    parser.add_argument('directory', help='path to folder containing train, test, val folders')
    return parser.parse_args()

def delete_existing_annotations():
    # Delete existing annotation files
    for subset in SUBSETS:
        try:
            os.remove(f'annotations_{subset}.txt')
        except:
            pass

def write_annotations(args):
    # Load the json labels
    with open(f'wlasl-uncompressed/{args.json_file}') as fin:
        videos = json.load(fin)

    # Create the annotation files
    for subset in SUBSETS:
        for video_id in os.listdir(subset):
            class_id = videos[video_id]['action'][0]
            directory = f'{subset}/{video_id}'
            frames = len([frame for frame in os.listdir(directory) if os.path.isfile(os.path.join(directory, frame))])
            with open(f'annotations_{subset}.txt', 'a') as fout:
                fout.write(f'{subset}/{video_id} {frames} {class_id}\n')

if __name__ == '__main__':
    args = load_args()
    os.chdir(args.directory)
    delete_existing_annotations()
    write_annotations(args)