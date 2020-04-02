import os
import datetime

from sklearn.metrics import precision_recall_curve, average_precision_score

import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter

import dtwalign

import numpy as np


import yaml

from pprint import pprint
from python_speech_features import mfcc
import scipy.io.wavfile as wav

from src.model import SiameseDeepLSTMNet
from src.loss import TripletNetLoss
from src.audio_processing import AudioPreprocessor, SpeechCommandsDataCollector, AudioPreprocessorMFCCDeltaDelta
from src.data_collector import SiameseSpeechCommandsDataCollector
from src.soft_dtw import SoftDTW


def train(config):
    #############################################################
    # Simple Lstm train script
    #############################################################

    """
    Initialize
    """

    wanted_words = config['wanted_words']

    model_settings = config
    model_settings['label_count'] = len(wanted_words) + 2

    preproc = AudioPreprocessorMFCCDeltaDelta(numcep=model_settings['dct_coefficient_count'],
                                              winlen=model_settings['winlen'],
                                              winstep=model_settings['winstep'])

    data_root = config['data_root']

    data_iter = SiameseSpeechCommandsDataCollector(preproc,
                                            data_dir=data_root,
                                            wanted_words=wanted_words,
                                            testing_percentage=10,
                                            validation_percentage=10
                                            )

    # Summary writer
    dt = str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p"))
    report_dir = os.path.join(config['report_root'], dt)
    writer = SummaryWriter(report_dir)

    with open(os.path.join(report_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    # create directory for model
    model_dir = os.path.join(config['model_root'], dt)
    os.makedirs(model_dir, exist_ok=True)

    # configure training procedure
    n_train_steps = config['train_steps']
    n_mini_batch_size = config['mini_batch_size']
    device = 'cuda' if torch.cuda.is_available() and config['use_cuda'] else 'cpu'


    siamese_net = SiameseDeepLSTMNet(config).to(device)
    siamese_net.train()
    optimizer = torch.optim.Adam(siamese_net.parameters(), lr=config['learning_rate'])
    loss_func = TripletNetLoss(config['loss'])

    """
    Train 
    """
    max_cce_acc = 0.
    for i in range(n_train_steps):
        # collect data
        data = data_iter.get_data(n_mini_batch_size, 0, 'training')
        labels = data['y']

        duplicates = data_iter.get_duplicates(labels, 0, 'training')
        assert np.any(labels == duplicates['y'])

        non_duplicates = data_iter.get_nonduplicates(labels, 0, 'training')
        assert np.any(labels != non_duplicates['y'])

        # construct a tensor of a form [data duplicates, data non-duplicates]
        x = torch.from_numpy(np.array([np.concatenate([data['x'], data['x']]),
                                       np.concatenate([duplicates['x'], non_duplicates['x']])])).float().to(device)
        y_target = np.array([0] * len(labels) + [1] * len(labels))
        y_target = torch.from_numpy(y_target).float().to(device)
        # forward
        embeds, y, predicted_labels = siamese_net(x)

        target_labels = torch.from_numpy(np.array([np.concatenate([data['y'],
                                                                   data['y'],
                                                                   duplicates['y'],
                                                                   non_duplicates['y']])]
                                                  )).long().to(device).squeeze()

        loss = loss_func(predicted_labels, target_labels, embeds, i)

        optimizer.zero_grad()

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(siamese_net.parameters(), 1)
        optimizer.step()

        loss_type = config['loss']['type']
        writer.add_scalar(f'{loss_type} loss', loss.detach().cpu().item(), i)

        if i % config['save_each'] == 0:
            for name, param in siamese_net.named_parameters():
                writer.add_histogram("SiameseNet_" + name, param, i)
            fname = os.path.join(model_dir, f'net_{i}.net')
            torch.save(siamese_net, fname)

        print(f"{i} | {loss_type} loss: {loss.detach().cpu().item():.4f}")

        if i % config['validate_each'] == 0:
            # collect data
            data = data_iter.get_data(n_mini_batch_size, 0, 'validation')
            labels = data['y']

            duplicates = data_iter.get_duplicates(labels, 0, 'validation')
            assert np.any(labels == duplicates['y'])

            non_duplicates = data_iter.get_nonduplicates(labels, 0, 'validation')
            assert np.any(labels != non_duplicates['y'])

            # construct a tensor of a form [data duplicates, data non-duplicates]
            x = torch.from_numpy(np.array([np.concatenate([data['x'], data['x']]),
                                           np.concatenate([duplicates['x'], non_duplicates['x']])])).float().to(device)
            y_target = np.array([0] * len(labels) + [1] * len(labels))
            # forward
            embeds, y, predicted_labels = siamese_net(x)

            target_labels = torch.from_numpy(np.array([np.concatenate([data['y'],
                                                                       data['y'],
                                                                       duplicates['y'],
                                                                       non_duplicates['y']])]
                                                      )).long().to(device).squeeze()
            val_loss = loss_func(predicted_labels, target_labels, embeds, i)

            # calculate distance between embeddings
            dists = []
            dist_func = SoftDTW(open_end=False, dist='l1')
            for k in range(embeds[0].shape[0]):
                dist = dist_func(embeds[0][k], embeds[1][k]).detach().cpu().item()
                dists.append(dist)

            dists = np.array(dists)

            ap = average_precision_score(y_target, dists)

            writer.add_scalar('AP', ap, i)
            writer.add_scalar(f'Validation loss {loss_type}', val_loss.detach().cpu().item(), i)

            print(f'validation {i}| AP: {ap:.4f} | {loss_type} loss: {val_loss.detach().cpu().item():.4f}')


if __name__ == '__main__':
    with open('../configs/experiment_0.yaml', 'r') as data_file:
        config = yaml.safe_load(data_file)
    pprint(config)

    train(config)



    print('done')
