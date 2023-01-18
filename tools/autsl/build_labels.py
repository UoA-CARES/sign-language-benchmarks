import csv
import os
import argparse

SUBSETS = ['train', 'test', 'val']


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help='path to folder rawframes')
    return parser.parse_args()


def delete_existing_annotations():
    # Delete existing annotation files
    for subset in SUBSETS:
        try:
            os.remove(f'{subset}_annotations.txt')
        except:
            pass


def write_annotations():
    # Create the annotation files
    for subset in SUBSETS:
        with open(f'{subset}_labels.csv', 'r') as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                video_id = row[0] + '_color'
                class_id = row[1]
                directory = f'rawframes/{subset}/{video_id}'

                try:
                    # Skip if the video is not found
                    frames = len([frame for frame in os.listdir(directory)
                                  if os.path.isfile(os.path.join(directory, frame))])

                    with open(f'{subset}_annotations.txt', 'a') as fout:
                        fout.write(
                            f'{subset}/{video_id} {frames} {class_id}\n')
                except:
                    print(f'{directory} not found. Skipping...')


if __name__ == '__main__':
    args = load_args()
    os.chdir(args.directory)
    delete_existing_annotations()
    write_annotations()
