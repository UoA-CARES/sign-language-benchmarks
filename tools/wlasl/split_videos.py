import json
import os
import argparse

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file', help='json annotation file name')
    parser.add_argument('directory', help='path to the uncompressed archive')
    return parser.parse_args()

def split_videos(args):
    # Load the json fle
    with open(args.json_file) as fin:
        videos = json.load(fin)

    # Create the folders for split videos
    os.makedirs('train', exist_ok=True)
    os.makedirs('test', exist_ok=True)
    os.makedirs('val', exist_ok=True)

    # Move the videos into respective folders
    for video_id in videos:
        subset = videos[video_id]['subset']
        try:
            os.rename(f'videos/{video_id}.mp4', f'{subset}/{video_id}.mp4')
        except FileNotFoundError:
            print(f'{video_id}.mp4 not found. Skipping and going next...')

if __name__ == '__main__':
    args = load_args()
    os.chdir(args.directory)
    split_videos(args)