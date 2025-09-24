import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

from utils.utils import load_yaml


class SystemDataset(TorchDataset):

    def __init__(self, scores, performances, pieces, args):

        self.scores = scores
        self.performances = performances
        self.pieces = pieces

        # for the snippets
        self.spec_context = int(args.snippet_len * args.fps)
        self.spec_bins = args.spec_bins
        self.sheet_context = args.sheet_context // 2
        self.staff_height = args.staff_height // 2

        self.sheet_overlap = args.sheet_overlap
        self.sheet_hop = int(self.sheet_context * (1 - self.sheet_overlap))
        self.sheet_left_pad = 20

        self.spec_overlap = args.spec_overlap
        self.spec_hop = int(self.spec_context * (1 - self.spec_overlap))

        self.system_translation = args.aug_configs[args.aug]['system_translation']
        self.sheet_scaling = args.aug_configs[args.aug]['sheet_scaling']
        self.onset_translation = args.aug_configs[args.aug]['onset_translation']

        self.train_entities = []
        self.prepare_train_entities()

    def prepare_train_entities(self):

        for i_score, score in tqdm(enumerate(self.scores), total=len(self.scores), ncols=80, leave=False):
            for i_spec, perf in enumerate(self.performances[i_score]):
                spectrogram, o2sc_maps = perf['spec'], perf['o2sc']
                edge_map = np.array([[score.shape[1], spectrogram.shape[1]]])
                o2sc_maps = np.concatenate((o2sc_maps, edge_map))
                for i_system, (start, stop) in enumerate(zip(o2sc_maps[:-1], o2sc_maps[1:])):
                    self.train_entities.append({'i_score': i_score, 'i_spec': i_spec, 'i_system': (start, stop)})

    def prepare_image_system(self, i_score, o2scs):

        sheet = self.scores[i_score]
        system_start, system_stop = o2scs[0][0], o2scs[1][0]
        system_image = sheet[:, system_start:system_stop]
        system_image = cv2.resize(system_image / 255, (system_image.shape[1] // 2, system_image.shape[0] // 2))

        snippet_starts = list(range(0, system_image.shape[1] + self.sheet_left_pad, self.sheet_hop))

        right_pad = max(0, snippet_starts[-1] + self.sheet_context - system_image.shape[1] - self.sheet_left_pad)
        right_pad_ones = np.ones((system_image.shape[0], right_pad), dtype=np.float32)
        left_pad_ones = np.ones((system_image.shape[0], self.sheet_left_pad), dtype=np.float32)
        system_image = np.hstack((left_pad_ones, system_image, right_pad_ones))
        snippets = np.array([system_image[10:90, t:t + self.sheet_context] for t in snippet_starts])
        return snippets

        # snippets = []
        # # simple snippet cutting, no augmentations
        # for x0 in snippet_starts:
        #     x1 = x0 + self.sheet_context
        #     y0 = system_image.shape[0] // 2 - self.staff_height // 2
        #     y1 = y0 + self.staff_height
        #     snippets.append(system_image[y0:y1, x0:x1])
        # return snippets

    def prepare_audio_system(self, i_score, i_spec, o2scs):

        full_spec = self.performances[i_score][i_spec]['spec']
        x0, x1 = o2scs[0][1], o2scs[1][1]
        snippet_starts = list(range(x0, x1, self.spec_hop))

        pad = max(0, snippet_starts[-1] + self.spec_context - full_spec.shape[1])
        x1 = min(snippet_starts[-1] + self.spec_context, full_spec.shape[1])
        system_spec = np.hstack((full_spec[:, x0:x1], np.zeros((full_spec.shape[0], pad), dtype=np.float32)))
        snippet_starts = [j - snippet_starts[0] for j in snippet_starts]
        snippets = np.array([system_spec[:, t:t + self.spec_context] for t in snippet_starts])
        return snippets

        # snippets = []
        # for x0 in snippet_starts:
        #     x1 = x0 + self.spec_context
        #     pad = max(0, x1 - full_spec.shape[1])
        #     snippets.append(np.hstack((full_spec[:, x0:x1], np.zeros((full_spec.shape[0], pad), dtype=np.float32))))
        # return snippets

    def __len__(self):
        return len(self.train_entities)

    def __getitem__(self, item):
        entity: dict = self.train_entities[item]
        i_score, i_spec, o2scs = entity['i_score'], entity['i_spec'], entity['i_system']

        sheet_fragment = self.prepare_image_system(i_score, o2scs)
        spec_fragment = self.prepare_audio_system(i_score, i_spec, o2scs)

        return sheet_fragment, spec_fragment

    def plot_statistics(self):

        o2scs = [t['i_system'] for t in self.train_entities]
        sys_audio_lens = [o[1][1] - o[0][1] for o in o2scs]
        sys_audio_lens = np.array(sys_audio_lens) / 20
        print(sys_audio_lens.shape)
        d = sys_audio_lens[sys_audio_lens >= 10]
        print(d.shape)

        import seaborn as sns
        sns.set_theme()
        sns.histplot(sys_audio_lens, bins=100)
        plt.xlim([0, 30])
        plt.xlabel("Duration (s)")
        plt.ylabel("Number of systems")
        plt.show()

    def get_durations(self):
        o2scs = [t['i_system'] for t in self.train_entities]
        sys_audio_lens = [o[1][1] - o[0][1] for o in o2scs]
        sys_audio_lens = np.array(sys_audio_lens) / 20
        return sys_audio_lens


def load_msmd_piece(path, piece, args, aug):

    npz_content = np.load(os.path.join(path, f'{piece}.npz'), allow_pickle=True)

    # check which performances match the augmentation pattern
    aug_config = args.aug_configs[aug]
    piece_valid_performances = []
    # t = 0.5
    # aug_config["tempo_range"] = [t, t]
    for perf in npz_content['performances']:
        tempo, synth = perf['perf'].split("tempo-")[1].split("_", 1)
        tempo = float(tempo) / 1000

        if synth not in aug_config["synths"] or tempo < aug_config["tempo_range"][0] \
                or tempo > aug_config["tempo_range"][1]:
            continue
        piece_valid_performances.append(perf)

    return npz_content['unrolled_score'], piece_valid_performances


def load_system_dataset(path, pieces, args, aug='full_aug'):

    scores = []
    performances = []
    for piece in tqdm(pieces, total=len(pieces), ncols=80, leave=False):
        unrolled_score, piece_valid_performances = load_msmd_piece(path, piece, args, aug)
        scores.append(unrolled_score)
        performances.append(piece_valid_performances)

    return SystemDataset(scores, performances, pieces, args)


def load_umc_dataset(path, pieces, args):

    scores = []
    performances = []
    for piece in tqdm(pieces, total=len(pieces), ncols=80, leave=False):
        npz_content = np.load(os.path.join(path, f'{piece}.npz'), allow_pickle=True)
        scores.append(npz_content['unrolled_score'])
        performances.append(npz_content['performances'])

    return SystemDataset(scores, performances, pieces, args)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    configs = load_yaml('config/msmd_config.yaml')
    for k, v in configs.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    prepared_data_path = '/home/luis/data/prepared_msmd'
    # prepared_data_path = '/home/luis/data/prepared_umc_alignments'
    split = load_yaml('splits/msmd_split.yaml')['train']
    # split = load_yaml('splits/db_scanned_recording.yaml')['test']

    dataset = load_system_dataset(prepared_data_path, split, parser.parse_args(), aug='no_aug')
    # dataset = load_umc_dataset(prepared_data_path, split, parser.parse_args())

    dataset.plot_statistics()
    print(len(dataset))
