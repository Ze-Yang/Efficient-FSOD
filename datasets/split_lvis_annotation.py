import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,
                        default="datasets/lvis",
                        help='path to the annotation file')
    parser.add_argument('--split', type=str, nargs='*',
                        default="lvis_v0.5_train".split(),
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
        }
        ids = [cat['id'] for cat in ann_train['categories'] if cat['frequency'] in s]
        ann_s['categories'] = [ann for ann in ann_train['categories'] if ann['id'] in ids]
        ann_s['annotations'] = [
            ann for ann in ann_train['annotations'] if ann['category_id'] in ids]
        img_ids = set([ann['image_id'] for ann in ann_s['annotations']])
        new_images = [img for img in ann_train['images'] if img['id'] in img_ids]
        ann_s['images'] = new_images

        save_path = os.path.join(save_dir, '{}_{}.json'.format(split, name))
        print('Saving {} annotations to {}.'.format(name, save_path))
        with open(save_path, 'w') as f:
            json.dump(ann_s, f)


if __name__ == '__main__':
    args = parse_args()
    for split in args.split:
        split_annotation(args.data, split, args.save_dir)
