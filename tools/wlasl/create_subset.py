import json
import os
import argparse

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file', help='json annotation file name')
    parser.add_argument('n_samples', help='number of samples per class', type=int)
    parser.add_argument('directory', help='path to the uncompressed archive')
    parser.add_argument('save_name', help='name of the subset json file')
    return parser.parse_args()

def load_json(filename):
    with open(filename) as fin:
        videos = json.load(fin)
    return videos

def create_subset(n_samples, videos):
    subset = {}
    sample_count = {}
    for video_id in videos:
        if videos[video_id]['subset'] == 'test':
            subset[video_id] = videos[video_id]
        else:
            class_id = videos[video_id]['action'][0]
            if class_id not in sample_count:
                sample_count[class_id] = 0      
                subset[video_id] = videos[video_id]      
                subset[video_id]['subset'] = 'train'
            elif sample_count[class_id] < n_samples-1:
                sample_count[class_id] = sample_count[class_id] + 1
                subset[video_id] = videos[video_id]
                subset[video_id]['subset'] = 'train'
    
    return subset

def save_json(subset, filename='new_subset.json'):
    with open(filename, 'w') as fin:
        json.dump(subset, fin)


if __name__ == '__main__':
    args = load_args()
    os.chdir(args.directory)
    videos = load_json(args.json_file)
    subset = create_subset(args.n_samples, videos)
    save_json(subset, args.save_name)
