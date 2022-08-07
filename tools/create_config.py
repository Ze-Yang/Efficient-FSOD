import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='voc', help='', choices=['voc'])
    parser.add_argument('--config_root', type=str, default='configs', help='the path to config dir')
    parser.add_argument('--split', type=int, default=1, help='only for voc')
    parser.add_argument('--method', type=str, default='ours', help='method to run', choices=['ours', 'defrcn'])
    parser.add_argument('--shot', type=int, default=1, help='shot to run experiments over')
    args = parser.parse_args()
    return args


def load_config_file(yaml_path):
    fpath = os.path.join(yaml_path)
    yaml_info = open(fpath).readlines()
    return yaml_info


def save_config_file(yaml_info, yaml_path):
    wf = open(yaml_path, 'w')
    for line in yaml_info:
        wf.write('{}'.format(line))
    wf.close()


def main():
    args = parse_args()

    if args.dataset == 'voc':
        name_template = 'faster_rcnn_R_101_C4_splitx_{}_{}shot.yaml'
        yaml_path = os.path.join(args.config_root, 'PascalVOC-Detection',
                                 name_template.format(args.method, args.shot))
        yaml_info = load_config_file(yaml_path)
        for i, lineinfo in enumerate(yaml_info):
            if '  TRAIN: ' in lineinfo:
                yaml_info[i] = lineinfo.format(args.split, args.split)
            if '  TEST: ' in lineinfo:
                yaml_info[i] = lineinfo.format(args.split)
        yaml_path = yaml_path.replace('splitx', 'split{}'.format(args.split))
    else:
        raise NotImplementedError

    save_config_file(yaml_info, yaml_path)


if __name__ == '__main__':
    main()
