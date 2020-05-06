from time import time
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn import datasets
from sklearn.manifold import TSNE
from itertools import chain
import pickle
import torch
import os

# parser = argparse.ArgumentParser(description='Vis')
# parser.add_argument('-l', '--load-name', default='gt_att_vectors_fullshot', type=str,
#                     help='')
# parser.add_argument('-s', '--save-name', default='tsne_plot', type=str,
#                     help='')
# parser.add_argument('--save-folder', default='results/vis', type=str,
#                     help='Dir to save results')
# parser.add_argument("--retest", action="store_true", help="test using the saved results")
# parser.add_argument("--save", action="store_true", help="save the results")
# args = parser.parse_args()

CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


def transform(data_dict, trim=False):
    # num = min([x.size(0) for x in data_dict.values()])
    num = 20
    if trim:
        data_dict = {i: x[:num]for i, x in data_dict.items()}
    data = [torch.sigmoid(x) / torch.sigmoid(x).mean(1, keepdim=True) for x in data_dict.values()]
    mean = torch.stack([x.mean(0) for x in data], dim=0).cpu().numpy()
    data = torch.cat(data, dim=0).cpu().numpy()
    label = np.array(list(chain.from_iterable([[int(i)] * x.size(0) for i, x in data_dict.items()])))
    mean_label = np.arange(20)
    return mean, mean_label, data, label


def data_prepare(gt_att, support, learned):
    gt_att_mean = (gt_att/gt_att.mean(1, keepdim=True)).cpu().numpy()
    gt_att_label_mean = np.arange(20)
    # gt_att_mean, gt_att_label_mean, gt_att_data, gt_att_label = transform(gt_att, True)
    support_mean, support_label_mean, support_data, support_label = transform(support)
    learned_data = learned.cpu().numpy()
    learned_label = np.arange(20)
    # data = np.vstack((gt_att_data, support_data, gt_att_mean, support_mean, learned_data))
    # label_list = [gt_att_label, support_label, gt_att_label_mean, support_label_mean, learned_label]
    data = np.vstack((support_data, gt_att_mean, support_mean, learned_data))
    label_list = [support_label, gt_att_label_mean, support_label_mean, learned_label]
    # data = np.vstack((support_data, support_mean))
    # label_list = [support_label, support_label_mean]
    num = np.cumsum([x.shape[0] for x in label_list])
    label = np.concatenate(label_list)

    return data, label, num


def get_data():
    digits = datasets.load_digits(n_class=6)
    data = digits.data
    label = digits.target
    n_samples, n_features = data.shape
    return data, label, n_samples, n_features


def plot(data, label, marker, title):
    # plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=label, s=30, marker=marker, cmap=plt.cm.rainbow)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    # plt.show()
    return


def plot_embedding(data, label, num, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    num = np.delete(num, -1)
    # gt_att, support, gt_att_mean, support_mean, learned = np.split(data, num, axis=0)
    # gt_att_lbl, support_lbl, gt_att_mean_lbl, support_mean_lbl, learned_lbl = np.split(label, num, axis=0)
    support, gt_att_mean, support_mean, learned = np.split(data, num, axis=0)
    support_lbl, gt_att_mean_lbl, support_mean_lbl, learned_lbl = np.split(label, num, axis=0)
    # support, support_mean = np.split(data, num, axis=0)
    # support_lbl, support_mean_lbl = np.split(label, num, axis=0)

    # plt.subplot(111)
    # for i in range(data.shape[0]):
        # plt.text(data[i, 0], data[i, 1], str(label[i]),
        #          color=plt.cm.Set1(label[i] / 20.),
        #          fontdict={'weight': 'bold', 'size': 9})
    # plot(support, support_lbl, '*', 'support_mean')
    plt.figure()
    # for i in range(support.shape[0]):
    #     plt.text(support[i, 0], support[i, 1], str(support_lbl[i]),
    #                       color=plt.cm.Set1(support_lbl[i] / 20.),
    #                       fontdict={'weight': 'bold', 'size': 9})
    # plot(support, support_lbl, 'o', 'support')
    plot(gt_att_mean, gt_att_mean_lbl, '*', 'gtt_mean')
    plot(support_mean, support_mean_lbl, 'o', 'support_mean')
    support_mean = support_mean + np.array([-0.03, 0.02])
    # support_mean[13] = support_mean[13] + np.array([-0.06, -0.05])
    support_mean[12] = support_mean[12] + np.array([0.02, 0])
    support_mean[1] = support_mean[1] + np.array([-0.02, 0])
    for i in range(support_mean.shape[0]):
        plt.text(support_mean[i, 0], support_mean[i, 1], CLASS_NAMES[support_mean_lbl[i]],
                          color=plt.cm.rainbow(support_mean_lbl[i] / 19.),
                          fontdict={'weight': 'bold', 'size': 9})
    # plt.show()
    #
    # plt.figure()

    # plot(learned, learned_lbl, '^', 'leaned')
    plt.show()

    # plt.scatter(gt_att_mean[:, 0], gt_att_mean[:, 1], c=gt_att_mean_lbl, s=180, marker='*', cmap=plt.cm.Spectral)
    # plt.scatter(support_mean[:, 0], support_mean[:, 1], c=support_mean_lbl, s=180, marker='o', cmap=plt.cm.Spectral)
    # plt.scatter(learned[:, 0], learned[:, 1], c=learned_lbl, s=180, marker='^', cmap=plt.cm.Spectral)
    # plt.xticks([])
    # plt.yticks([])
    # plt.title(title)
    # plt.show()
    # plt.scatter(data[:, 0], data[:, 1], c=label, s=180, cmap=plt.cm.Spectral)
    return





def main():
    if args.retest:
        result_file = os.path.join(args.save_folder, args.save_name + '.pkl')
        f = open(result_file, 'rb')
        result, label, num = pickle.load(f)
        plot_embedding(result, label, num,
                       't-SNE embedding of the digits')
        return
    # file = os.path.join(args.save_folder, args.load_name+'.pkl')
    file = os.path.join(args.save_folder, 'learned_fullshot.pkl')
    f = open(file, 'rb')
    gt_att = pickle.load(f)
    file = os.path.join(args.save_folder, 'learned_10shot_in.pkl')
    f = open(file, 'rb')
    learned = pickle.load(f)
    file = os.path.join(args.save_folder, 'support_10shot_in.pkl')
    f = open(file, 'rb')
    support = pickle.load(f)
    data, label, num = data_prepare(gt_att, support, learned)
    # data, label, n_samples, n_features = get_data()
    print('Computing t-SNE embedding...')
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    print('Done.')
    if args.save:
        result_file = os.path.join(args.save_folder, args.save_name+'.pkl')
        with open(result_file, 'wb') as f:
            pickle.dump((result, label, num), f, pickle.HIGHEST_PROTOCOL)
    plot_embedding(result, label, num,
                     't-SNE embedding of the digits (time %.2fs)'
                     % (time() - t0))


if __name__ == '__main__':
    main()
