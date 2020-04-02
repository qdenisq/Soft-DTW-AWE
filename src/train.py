import os
import datetime

import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter

import dtwalign

import numpy as np
from six.moves import xrange

import yaml

from pprint import pprint
from python_speech_features import mfcc
import scipy.io.wavfile as wav

from src.audio_processing import AudioPreprocessor, SpeechCommandsDataCollector, AudioPreprocessorMFCCDeltaDelta
from src.soft_dtw import SoftDTW


class SiameseSpeechCommandsDataCollector(SpeechCommandsDataCollector):
    def get_duplicates(self, labels, offset, mode):
        # Pick one of the partitions to choose samples from.
        candidates = self.data_index[mode]
        how_many = len(labels)
        duplicate_labels = np.zeros(len(labels))
        if how_many == -1:
            sample_count = len(candidates)
        else:
            sample_count = max(0, min(how_many, len(candidates) - offset))
        # Data and labels will be populated and returned.
        data = []
        seq_lens = []
        # labels = np.zeros(sample_count)
        pick_deterministically = False
        # Use the processing graph we created earlier to repeatedly to generate the
        # final output sample data we'll use in training.
        for i in xrange(offset, offset + sample_count):
            # Pick which audio sample to use.
            if how_many == -1 or pick_deterministically:
                sample_index = i
            else:
                while True:
                    sample_index = np.random.randint(len(candidates))
                    if self.word_to_index[candidates[sample_index]['label']] == labels[i]:
                        break
            sample = candidates[sample_index]

            sample_data = self.get_sequential_data_sample(sample['file'])
            seq_len = sample_data.shape[0]
            data.append(sample_data)
            seq_lens.append(seq_len)
            label_index = self.word_to_index[sample['label']]
            duplicate_labels[i - offset] = label_index
        max_seq_len = max(seq_lens)
        zero_padded_data = [np.append(s, np.zeros((max_seq_len - s.shape[0], s.shape[1])), axis=0) for s in data]
        data = np.stack(zero_padded_data)
        seq_lens = np.array(seq_lens)
        duplicate_labels = np.array(duplicate_labels)
        return {'x': data,
                'y': duplicate_labels,
                'seq_len': seq_lens}

    def get_nonduplicates(self, labels, offset, mode):
        # Pick one of the partitions to choose samples from.
        candidates = self.data_index[mode]
        how_many = len(labels)
        nonduplicate_labels = np.zeros(len(labels))
        if how_many == -1:
            sample_count = len(candidates)
        else:
            sample_count = max(0, min(how_many, len(candidates) - offset))
        # Data and labels will be populated and returned.
        data = []
        seq_lens = []
        # labels = np.zeros(sample_count)
        pick_deterministically = False
        # Use the processing graph we created earlier to repeatedly to generate the
        # final output sample data we'll use in training.
        for i in xrange(offset, offset + sample_count):
            # Pick which audio sample to use.
            if how_many == -1 or pick_deterministically:
                sample_index = i
            else:
                while True:
                    sample_index = np.random.randint(len(candidates))
                    if self.word_to_index[candidates[sample_index]['label']] != labels[i]:
                        break
            sample = candidates[sample_index]

            sample_data = self.get_sequential_data_sample(sample['file'])
            seq_len = sample_data.shape[0]
            data.append(sample_data)
            seq_lens.append(seq_len)
            label_index = self.word_to_index[sample['label']]
            nonduplicate_labels[i - offset] = label_index
        max_seq_len = max(seq_lens)
        zero_padded_data = [np.append(s, np.zeros((max_seq_len - s.shape[0], s.shape[1])), axis=0) for s in data]
        data = np.stack(zero_padded_data)
        seq_lens = np.array(seq_lens)
        duplicate_labels = np.array(nonduplicate_labels)
        return {'x': data,
                'y': duplicate_labels,
                'seq_len': seq_lens}


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class SiameseDeepLSTMNet(nn.Module):
    def __init__(self, model_settings):
        super(SiameseDeepLSTMNet, self).__init__()
        self.__n_window_height = model_settings['mfcc_num']
        self.__n_classes = model_settings['label_count']

        self.__dropout = nn.Dropout(p=model_settings['dropout'])

        if 'batch_norm' in model_settings and model_settings['batch_norm'] == True:
            self.__bn1 = nn.BatchNorm1d(self.__n_window_height)

        self.__n_hidden_reccurent = model_settings['hidden_reccurent']
        self.__n_hidden_fc = model_settings['hidden_fc']
        if 'bidirectional' in model_settings:
            bidirectional = model_settings['bidirectional']
        else:
            bidirectional = False
        k = 2 if bidirectional else 1
        self.lstms = nn.ModuleList([nn.LSTM(self.__n_window_height, self.__n_hidden_reccurent[0], batch_first=True, bidirectional=bidirectional)])
        self.lstms.extend(
            [nn.LSTM(self.__n_hidden_reccurent[i-1]*k, self.__n_hidden_reccurent[i], batch_first=True, bidirectional=bidirectional)
             for i in range(1, len(self.__n_hidden_reccurent))])

        if self.__n_hidden_fc is not None:
            self.linears = nn.ModuleList(
                [nn.Linear(self.__n_hidden_reccurent[-1]*k, self.__n_hidden_fc[0])])
            self.linears.extend(
                [nn.Linear(self.__n_hidden_fc[i-1], self.__n_hidden_fc[i]) for i in range(1, len(self.__n_hidden_fc))])

            self.__output_layer = nn.Linear(self.__n_hidden_fc[-1] * 2, 1)
            self.__output_layer_cce = nn.Linear(self.__n_hidden_fc[-1], self.__n_classes)
        else:
            self.__output_layer = nn.Linear(self.__n_hidden_reccurent[-1] * 2 * k, 1)
            self.__output_layer_cce = nn.Linear(self.__n_hidden_reccurent[-1] * k, self.__n_classes)
        
        # self.apply(init_weights)

    def single_forward(self, input, hidden=None):
        x = input
        if hasattr(self, '__bn1'):
            orig_shape = x.shape
            x = x.view(-1, self.__n_window_height)
            x = self.__bn1(x)
            x = x.view(orig_shape)
        x = self.__dropout(x)

        for i in range(len(self.__n_hidden_reccurent) - 1):
            x, hidden = self.lstms[i](x, None)
            x = self.__dropout(x)
        x, hidden = self.lstms[-1](x, None)

        if self.__n_hidden_fc is not None:
            for i in range(len(self.__n_hidden_fc)-1):
                x = torch.relu(self.linears[i](x))
                x = self.__dropout(x)
            # x = torch.tanh(self.linears[-1](x))
            x = self.linears[-1](x)
        hidden = x[:, -1, :]
        return x, hidden

    def forward(self, input, hidden=None):
        # if hidden is None:
        #     hidden = torch.zeros(x.size(0), self.n_hidden)
        lstm_out = []
        zs = []
        for i in range(2):
            x = input[i]
            x, hidden = self.single_forward(x)
            lstm_out.append(hidden)
            zs.append(x)

        # cce path
        cce_output = torch.cat(lstm_out, dim=0).squeeze()
        cce_output = self.__output_layer_cce(cce_output)

        # bce path
        lstm_out = torch.cat(lstm_out, dim=-1).squeeze()
        output = torch.nn.Sigmoid()(self.__output_layer(lstm_out))
        return zs, output, cce_output


def anneal_function(anneal_func, step, k, x0):
    if anneal_func == 'logistic':
        return float(1 / (1 + np.exp(-k * (step - x0))))
    elif anneal_func == 'linear':
        return min(1, step / x0)


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels, axis=None) / float(labels.size)


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

    open_end = model_settings['open_end']
    dist = model_settings['dist']
    margin = model_settings['margin']

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
    model_dir = os.path.join(config['save_dir'], dt)
    os.makedirs(model_dir, exist_ok=True)

    # configure training procedure
    n_train_steps = config['train_steps']
    n_mini_batch_size = config['mini_batch_size']

    siamese_net = SiameseDeepLSTMNet(model_settings).to('cuda')
    siamese_net.train()
    optimizer = torch.optim.Adam(siamese_net.parameters(), lr=config['learning_rate'])

    soft_dtw_loss = SoftDTW(open_end=open_end, dist=dist)

    # dummy_input = torch.zeros(2, 1, 25, 26).cuda()
    # writer.add_graph(siamese_net, dummy_input)
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
                                       np.concatenate([duplicates['x'], non_duplicates['x']])])).float().to('cuda')
        y_target = np.array([0] * len(labels) + [1] * len(labels))
        y_target = torch.from_numpy(y_target).float().to('cuda')
        # forward
        zs, y, predicted_labels = siamese_net(x)

        #
        target_labels = torch.from_numpy(np.array([np.concatenate([data['y'],
                                                                   data['y'],
                                                                   duplicates['y'],
                                                                   non_duplicates['y']])]
                                                  )).long().to('cuda').squeeze()



        # bce loss
        bce_loss = nn.BCELoss()(y, y_target)

        # ce loss
        ce_loss = torch.nn.CrossEntropyLoss()(predicted_labels, target_labels)

        Triplet_loss_weight = anneal_function('logistic', i, config['triplet_anneal_k'], config['triplet_anneal_b'])
        Triplet_loss = torch.tensor([0]).float().cuda()

        alpha = config['alpha']

        if config['loss_type'] == 'sdtw':

            # DTWLoss (want to minimize dtw between duplica)
            DTW_loss = torch.tensor([0]).float().cuda()
            for k in range(n_mini_batch_size):
                DTW_loss += torch.nn.functional.relu(soft_dtw_loss(zs[0][k], zs[1][k]) -
                                                     soft_dtw_loss(zs[0][k + n_mini_batch_size], zs[1][k + n_mini_batch_size])
                                                     + margin)

            if open_end:
                DTW_loss /= (n_mini_batch_size)
            else:
                # DTW_loss /= (n_mini_batch_size * zs[0].shape[1])
                DTW_loss /= (n_mini_batch_size)
            Triplet_loss = DTW_loss
            loss = alpha * ce_loss + (1. - alpha) * Triplet_loss * Triplet_loss_weight

        elif config['loss_type'] == 'l2':
            L2_loss = torch.tensor([0]).float().cuda()

            L2_loss = torch.nn.functional.relu(torch.sum((zs[0][:n_mini_batch_size, -1, :] - zs[1][:n_mini_batch_size, -1, :])**2, dim=-1) -
                                               torch.sum((zs[0][:n_mini_batch_size, -1, :] - zs[1][n_mini_batch_size:, -1, :]) ** 2, dim=-1) +
                                                    margin).sum()
            L2_loss /= n_mini_batch_size
            Triplet_loss = L2_loss
            loss = alpha * ce_loss + (1. - alpha) * Triplet_loss * Triplet_loss_weight

        elif config['loss_type'] == 'cos_hinge':
            Cos_hinge_loss = torch.tensor([0]).float().cuda()

            Cos_hinge_loss = torch.clamp_min(
                    - torch.nn.CosineSimilarity(dim=-1)(zs[0][:n_mini_batch_size, -1, :], zs[1][:n_mini_batch_size, -1, :])
                    + torch.nn.CosineSimilarity(dim=-1)(zs[0][:n_mini_batch_size, -1, :], zs[1][n_mini_batch_size:, -1, :])
                    + margin, 0).sum()

            Cos_hinge_loss += torch.clamp_min(
                    - torch.nn.CosineSimilarity(dim=-1)(zs[0][:n_mini_batch_size, -1, :], zs[1][:n_mini_batch_size, -1, :])
                    + torch.nn.CosineSimilarity(dim=-1)(zs[1][:n_mini_batch_size, -1, :], zs[1][n_mini_batch_size:, -1, :])
                    + margin, 0).sum()


            # for k in range(n_mini_batch_size):
            #     Cos_hinge_loss += torch.nn.functional.relu(
            #         - torch.nn.CosineSimilarity(dim=0)(zs[0][k, -1, :], zs[1][k, -1, :]) +
            #         torch.nn.CosineSimilarity(dim=0)(zs[0][k + n_mini_batch_size, -1, :],
            #                                          zs[1][k + n_mini_batch_size, -1, :])
            #         + margin)

            Cos_hinge_loss /= (2*n_mini_batch_size)
            Triplet_loss = Cos_hinge_loss
            loss = alpha * ce_loss + (1. - alpha) * Cos_hinge_loss * Triplet_loss_weight
        elif config['loss_type'] == 'ce':
            loss = ce_loss
        else:
            raise KeyError(f"Unknown loss type: {config['loss_type']}")



        # loss = bce_loss + ce_loss + KL_loss * KL_weight
        # loss = ce_loss
        # loss = 0.2 * ce_loss + 0.8 * Triplet_loss * Triplet_loss_weight
        # backward and update
        optimizer.zero_grad()

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(siamese_net.parameters(), 1)
        optimizer.step()

        cce_acc = accuracy(predicted_labels.detach().cpu().numpy(), target_labels.detach().cpu().numpy())
        writer.add_scalar('CCE', ce_loss.detach().cpu(), i)
        writer.add_scalar('Triplet Loss', Triplet_loss.detach().cpu(), i)
        writer.add_scalar('CCE_Accuracy', cce_acc, i)
        writer.add_scalar('Triplet_loss_weight', Triplet_loss_weight, i)

        if i % 500 == 0:
            for name, param in siamese_net.named_parameters():
                writer.add_histogram("SiameseNet_" + name, param, i)

        if cce_acc > max_cce_acc or i % 500 == 0:
            if cce_acc > max_cce_acc:
                max_cce_acc = cce_acc
            fname = os.path.join(model_dir, f'net_{i}_{cce_acc}.net')
            torch.save(siamese_net, fname)

        print(f"{i} "
              f"| L: {loss.detach().cpu().numpy()} "
              f"| CE {ce_loss.detach().cpu().numpy()} "
              f"| Triplet Loss {Triplet_loss.detach().cpu()}")

        # TODO: add validation each 1k steps for example
        if i % 500 == 0:
            # collect data
            data = data_iter.get_data(n_mini_batch_size, 0, 'validation')
            labels = data['y']

            duplicates = data_iter.get_duplicates(labels, 0, 'validation')
            assert np.any(labels == duplicates['y'])

            non_duplicates = data_iter.get_nonduplicates(labels, 0, 'validation')
            assert np.any(labels != non_duplicates['y'])

            # construct a tensor of a form [data duplicates, data non-duplicates]
            x = torch.from_numpy(np.array([np.concatenate([data['x'], data['x']]),
                                           np.concatenate([duplicates['x'], non_duplicates['x']])])).float().to('cuda')
            y_target = np.array([0] * len(labels) + [1] * len(labels))
            y_target = torch.from_numpy(y_target).float().to('cuda')
            # forward
            zs, y, predicted_labels = siamese_net(x)

            #
            target_labels = torch.from_numpy(np.array([np.concatenate([data['y'],
                                                                       data['y'],
                                                                       duplicates['y'],
                                                                       non_duplicates['y']])]
                                                      )).long().to('cuda').squeeze()

            # backward and update
            # bce loss
            bce_loss = nn.BCELoss()(y, y_target)

            # CrossEntropy loss
            ce_loss = torch.nn.CrossEntropyLoss()(predicted_labels, target_labels)

            # KL divergence regularization
            loss = bce_loss + ce_loss

            y = np.array([[1.0 - v, v] for v in y.detach().cpu().numpy()]).squeeze()

            cce_acc = accuracy(predicted_labels.detach().cpu().numpy(), target_labels.detach().cpu().numpy())
            bce_acc = accuracy(y, y_target.detach().cpu().numpy())
            writer.add_scalar('valid_BCE', bce_loss.detach().cpu(), i)
            writer.add_scalar('valid_CCE', ce_loss.detach().cpu(), i)
            writer.add_scalar('valid_CCE_Accuracy', cce_acc, i)
            writer.add_scalar('valid_BCE_Accuracy', bce_acc, i)

            print('validation: ', i, loss.detach().cpu(), bce_loss.detach().cpu(), ce_loss.detach().cpu())


if __name__ == '__main__':
    with open('configs/experiment_0.yaml', 'r') as data_file:
        config = yaml.safe_load(data_file)
    pprint(config)

    train(config)



    print('done')
