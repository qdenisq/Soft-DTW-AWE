import os
import datetime
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import average_precision_score
import numpy as np
import yaml
from pprint import pprint

from src.model import SiameseDeepLSTMNet
from src.loss import TripletNetLoss
from src.audio_processing import AudioPreprocessorMFCCDeltaDelta
from src.data_collector import SiameseSpeechCommandsDataCollector
from src.soft_dtw import SoftDTW


def train(config):
    """Main routine for training triplet network on SpeechCommands dataset

    Parameters
    ----------
    config

    Returns
    -------

    """

    labels = config['labels']
    config['model']['label_count'] = len(labels) + 2

    preproc = AudioPreprocessorMFCCDeltaDelta(numcep=config['preprocessing']['numcep'],
                                              winlen=config['preprocessing']['winlen'],
                                              winstep=config['preprocessing']['winstep'],
                                              target_sample_rate=config['preprocessing']['target_sample_rate'])

    data_root = config['data_root']

    data_iter = SiameseSpeechCommandsDataCollector(preproc,
                                                    data_dir=data_root,
                                                    wanted_words=labels,
                                                    testing_percentage=10,
                                                    validation_percentage=10
                                                    )

    # Create summary writer
    dt = str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p"))
    report_dir = os.path.join(config['report_root'], dt)
    writer = SummaryWriter(report_dir)

    with open(os.path.join(report_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    # create directory for model
    model_dir = os.path.join(config['model_root'], dt)
    os.makedirs(model_dir, exist_ok=True)

    # configure training procedure
    n_train_steps = config['training']['train_steps']
    n_mini_batch_size = config['training']['mini_batch_size']
    device = 'cuda' if torch.cuda.is_available() and config['use_cuda'] else 'cpu'

    # init model
    siamese_net = SiameseDeepLSTMNet(config['model']).to(device)
    siamese_net.train()
    # init optimizer
    optimizer = torch.optim.Adam(siamese_net.parameters(), lr=config['training']['learning_rate'])
    # init loss
    loss_func = TripletNetLoss(config['loss'])

    """
    Train  loop
    """
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
        # forward
        embeds, predicted_labels = siamese_net(x)

        target_labels = torch.from_numpy(np.array([np.concatenate([data['y'],
                                                                   data['y'],
                                                                   duplicates['y'],
                                                                   non_duplicates['y']])]
                                                  )).long().to(device).squeeze()
        # compute loss
        loss = loss_func(predicted_labels, target_labels, embeds, i)

        # backprop and update
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(siamese_net.parameters(), 1)
        optimizer.step()

        loss_type = config['loss']['type']
        writer.add_scalar(f'{loss_type} loss', loss.detach().cpu().item(), i)

        if i % config['training']['save_each'] == 0:
            for name, param in siamese_net.named_parameters():
                writer.add_histogram("SiameseNet_" + name, param, i)
            fname = os.path.join(model_dir, f'net_{i}.pt')
            torch.save(siamese_net.state_dict(), fname)

        print(f"{i} | {loss_type} loss: {loss.detach().cpu().item():.4f}")

        # validate
        if i % config['training']['validate_each'] == 0:
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
            embeds, predicted_labels = siamese_net(x)

            target_labels = torch.from_numpy(np.array([np.concatenate([data['y'],
                                                                       data['y'],
                                                                       duplicates['y'],
                                                                       non_duplicates['y']])]
                                                      )).long().to(device).squeeze()

            # compute validation loss
            val_loss = loss_func(predicted_labels, target_labels, embeds, i)

            # compute distance between embeddings
            dists = []
            dist_func = SoftDTW(open_end=False, dist='l1')
            for k in range(embeds[0].shape[0]):
                dist = dist_func(embeds[0][k], embeds[1][k]).detach().cpu().item()
                dists.append(dist)
            dists = np.array(dists)

            # compute average precision
            ap = average_precision_score(y_target, dists)

            writer.add_scalar('AP', ap, i)
            writer.add_scalar(f'Validation loss {loss_type}', val_loss.detach().cpu().item(), i)

            print(f'validation {i}| AP: {ap:.4f} | {loss_type} loss: {val_loss.detach().cpu().item():.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to the config file", default='../configs/experiment_0.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)
    pprint(config)
    train(config)
    print('done')
