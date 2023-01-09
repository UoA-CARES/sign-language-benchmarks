import os
import pandas as pd
import json
import argparse

SUBSETS = ['train', 'test', 'val']

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('n_classes', help='number of classes', type=int)
    parser.add_argument(
        'directory', help='path to folder containing train, test, val folders')
    return parser.parse_args()

def delete_existing_annotations():
    # Delete existing annotation files
    for subset in SUBSETS:
        try:
            os.remove(f'{subset}_annotations.txt')
        except:
            pass

def fix_missing():
    ''' Read and delete the missing annotations from the nslt_2000.json file.

    Returns:
       dict: The cleaned annotations for the videos. 
    '''
    with open('missing.txt') as fin:
        missing = fin.readlines()

    with open('nslt_2000.json') as fin:
        videos = json.load(fin)

    for value in missing:
        try:
            videos.pop(value.strip('\n'))
        except KeyError:
            pass

    return videos


def get_topk(videos, k=10):
    ''' Get the top k classes with the most samples

    Args:
        videos (dict): The annotations for the videos.
        k (int): The number of classes to extract. Default: 10.
    '''
    df = pd.DataFrame(videos).transpose()
    df['class'] = df.apply(lambda x: x.action[0], axis=1)
    top10 = df['class'].value_counts().iloc[:k].index.tolist()
    for value in list(videos.keys()):
        if int(videos[value]['action'][0]) not in top10:
            try:
                videos.pop(value.strip('\n'))
            except KeyError:
                pass

    return videos


def save_json(videos, k=10):
    ''' Save a dictionary as a JSON file.

    Args:
        videos (dict): The annotations for the videos.
        k (int): The number of classes. Default: 10.
    '''
    with open(f'wlasl_{k}.json', 'w') as fout:
        fout.write(json.dumps(videos))


if __name__ == '__main__':
    # Make sure that this script is run after the wlasl file is extracted
    args = load_args()
    os.chdir(args.directory)
    delete_existing_annotations()
    videos = fix_missing()
    videos = get_topk(videos, args.n_classes)
    save_json(videos, args.n_classes)
