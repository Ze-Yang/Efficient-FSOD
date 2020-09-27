import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,
                        default="datasets/lvis",
                        help='path to the annotation file')
    parser.add_argument('--split', type=str, nargs='*',
                        default="lvis_v0.5_train lvis_v1_train lvis_v0.5_val lvis_v1_val".split(),
                        help='path to the annotation file')
    parser.add_argument('--save-dir', type=str,
                        default="datasets/lvis",
                        help='path to the save directory')
    args = parser.parse_args()
    return args


def split_annotation(data_path, split, save_dir):
    with open(os.path.join(data_path, split + '.json')) as f:
        ann_train = json.load(f)

    for s, name in [(('f', 'c'), 'base'), (('r',), 'novel')]:
        ann_s = {
            'info': ann_train['info'],
            'licenses': ann_train['licenses'],
            # We use the whole original dataset images, though some of them may contain no annotations
            # after the split operation. In other words, the images that do not contain target category
            # object should also be included, as they will be used to penalize the false positive
            # predictions from the detector during inference. As for training, these images will be
            # automatically filtered out in the data construction procedure.
            'images': ann_train['images'],
        }
        ids = [cat['id'] for cat in ann_train['categories'] if cat['frequency'] in s]
        ann_s['categories'] = [ann for ann in ann_train['categories'] if ann['id'] in ids]
        ann_s['annotations'] = [
            ann for ann in ann_train['annotations'] if ann['category_id'] in ids]

        save_path = os.path.join(save_dir, '{}_{}.json'.format(split, name))
        print('Saving {} annotations to {}.'.format(name, save_path))
        with open(save_path, 'w') as f:
            json.dump(ann_s, f)


if __name__ == '__main__':
    args = parse_args()
    for split in args.split:
        split_annotation(args.data, split, args.save_dir)
