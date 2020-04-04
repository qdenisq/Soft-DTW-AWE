from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import math
import os.path
import random
import re
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
from numpy.lib.stride_tricks import as_strided

import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav
from six.moves import xrange

from src.audio_processing import AudioPreprocessor, AudioPreprocessorMFCCDeltaDelta

RANDOM_SEED = 0
MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'

np.random.seed(RANDOM_SEED)


def prepare_words_list(wanted_words):
    """Prepends common tokens to the custom word list.

    Parameters
    ----------
    wanted_words: list
     List of strings containing the custom words.

    Returns
    -------
    list
        List with the standard silence and unknown tokens added.
    """

    return [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_words


def which_set(filename, validation_percentage, testing_percentage):
    """Determines which data partition the file should belong to.

    We want to keep files in the same training, validation, or testing sets even
    if new ones are added over time. This makes it less likely that testing
    samples will accidentally be reused in training when long runs are restarted
    for example. To keep this stability, a hash of the filename is taken and used
    to determine which set it should belong to. This determination only depends on
    the name and the set proportions, so it won't change as other files are added.

    It's also useful to associate particular files as related (for example words
    spoken by the same person), so anything after '_nohash_' in a filename is
    ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
    'bobby_nohash_1.wav' are always in the same set, for example.

    Parameters
    ----------
    filename:
        File path of the data sample.
    validation_percentage:  float
        How much of the data set to use for validation. (between 0 and 1)
    testing_percentage: float
        How much of the data set to use for testing. (between 0 and 1)

    Returns
    -------
    String
        one of 'training', 'validation', or 'testing'.
    """

    base_name = os.path.basename(filename)
    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of
    # grouping wavs that are close variations of each other.
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
    hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
        result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'testing'
    else:
        result = 'training'
    return result


class SpeechCommandsDataCollector(object):
    """Base class for Speech Commands dataset data iterator
    """
    def __init__(self,
                 audio_processor,
                 data_dir,
                 wanted_words,
                 validation_percentage,
                 testing_percentage,
                 silence_percentage=0.,
                 unknown_percentage=0.):
        random.seed(RANDOM_SEED)
        self.__audio_preproc = audio_processor
        self.data_dir = data_dir
        self.data_index = {"train": [],
                           "test": [],
                           "validation": []}
        self.prepare_data_index(silence_percentage=silence_percentage,
                                unknown_percentage=unknown_percentage,
                                wanted_words=wanted_words,
                                validation_percentage=validation_percentage,
                                testing_percentage=testing_percentage)
        return

    def prepare_data_index(self, silence_percentage, unknown_percentage,
                           wanted_words, validation_percentage,
                           testing_percentage):
        """Prepares a list of the samples organized by set and label.

        The training loop needs a list of all the available data, organized by
        which partition it should belong to, and with ground truth labels attached.
        This function analyzes the folders below the `data_dir`, figures out the
        right
        labels for each file based on the name of the subdirectory it belongs to,
        and uses a stable hash to assign it to a data set partition.

        Parameters
        ----------
          silence_percentage: float
            How much of the resulting data should be background.
          unknown_percentage: float
            How much should be audio outside the wanted classes.
          wanted_words: list
            Labels of the classes we want to be able to recognize.
          validation_percentage: float
            How much of the data set to use for validation.
          testing_percentage: float
            How much of the data set to use for testing.
        Returns
        -------
        dict
            Dictionary containing a list of file information for each set partition,
            and a lookup map for each class to determine its numeric index.
        """

        # Make sure the shuffling and picking of unknowns is deterministic.
        random.seed(RANDOM_SEED)
        wanted_words_index = {}
        for index, wanted_word in enumerate(wanted_words):
            wanted_words_index[wanted_word] = index + 2
        self.data_index = {'validation': [], 'testing': [], 'training': []}
        unknown_index = {'validation': [], 'testing': [], 'training': []}
        all_words = {}
        # Look through all the subfolders to find audio samples
        search_path = os.path.join(self.data_dir, '*', '*.wav')
        for wav_path in gfile.Glob(search_path):
            _, word = os.path.split(os.path.dirname(wav_path))
            word = word.lower()
            # Treat the '_background_noise_' folder as a special case, since we expect
            # it to contain long audio samples we mix in to improve training.
            if word == BACKGROUND_NOISE_DIR_NAME:
                continue
            all_words[word] = True
            set_index = which_set(wav_path, validation_percentage, testing_percentage)
            # If it's a known class, store its detail, otherwise add it to the list
            # we'll use to train the unknown label.
            if word in wanted_words_index:
                self.data_index[set_index].append({'label': word, 'file': wav_path})
            else:
                unknown_index[set_index].append({'label': word, 'file': wav_path})
        if not all_words:
            raise Exception('No .wavs found at ' + search_path)
        for index, wanted_word in enumerate(wanted_words):
            if wanted_word not in all_words:
                raise Exception('Expected to find ' + wanted_word +
                                ' in labels but only found ' +
                                ', '.join(all_words.keys()))
        # We need an arbitrary file to load as the input for the silence samples.
        # It's multiplied by zero later, so the content doesn't matter.
        silence_wav_path = self.data_index['training'][0]['file']
        for set_index in ['validation', 'testing', 'training']:
            set_size = len(self.data_index[set_index])
            silence_size = int(math.ceil(set_size * silence_percentage / 100))
            for _ in range(silence_size):
                self.data_index[set_index].append({
                    'label': SILENCE_LABEL,
                    'file': silence_wav_path
                })
            # Pick some unknowns to add to each partition of the data set.
            random.shuffle(unknown_index[set_index])
            unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
            self.data_index[set_index].extend(unknown_index[set_index][:unknown_size])
        # Make sure the ordering is random.
        for set_index in ['validation', 'testing', 'training']:
            random.shuffle(self.data_index[set_index])
        # Prepare the rest of the result data structure.
        self.words_list = prepare_words_list(wanted_words)
        self.word_to_index = {}
        for word in all_words:
            if word in wanted_words_index:
                self.word_to_index[word] = wanted_words_index[word]
            else:
                self.word_to_index[word] = UNKNOWN_WORD_INDEX
        self.word_to_index[SILENCE_LABEL] = SILENCE_INDEX

    def set_size(self, mode):
        """Calculates the number of samples in the dataset partition.

        Parameters
        ----------
        mode: str
            Which partition, must be 'training', 'validation', or 'testing'.

        Returns
        -------
        int
            Number of samples in the partition.
        """
        return len(self.data_index[mode])

    def get_sequential_data_sample(self, fname):
        """Loads data sample from file and performs predefined transformation

        Parameters
        ----------
        fname: str
            path to data sample

        Returns
        -------
        object
            Transformed data sample.

        """
        return self.__audio_preproc(fname)

    def get_data(self, how_many, offset, mode):
        """Gather samples from the data set, applying transformations as needed.

        When the mode is 'training', a random selection of samples will be returned,
        otherwise the first N clips in the partition will be used. This ensures that
        validation always uses the same samples, reducing noise in the metrics.

        Parameters
        ----------
          how_many: int
            Desired number of samples to return. -1 means the entire
            contents of this partition.
          offset: int
            Where to start when fetching deterministically.
          mode: str
            Which partition to use, must be 'training', 'validation', or
            'testing'.

        Returns
        -------
        dict
          List of sample data for the transformed samples, and list of label indexes, along with length of the data
        """

        # Pick one of the partitions to choose samples from.
        candidates = self.data_index[mode]
        if how_many == -1:
            sample_count = len(candidates)
        else:
            sample_count = max(0, min(how_many, len(candidates) - offset))
        # Data and labels will be populated and returned.
        data = []
        seq_lens = []
        labels = np.zeros(sample_count)
        pick_deterministically = (mode != 'training')
        # Use the processing graph we created earlier to repeatedly to generate the
        # final output sample data we'll use in training.
        for i in xrange(offset, offset + sample_count):
            # Pick which audio sample to use.
            if how_many == -1 or pick_deterministically:
                sample_index = i
            else:
                sample_index = np.random.randint(len(candidates))
            sample = candidates[sample_index]

            sample_data = self.get_sequential_data_sample(sample['file'])
            seq_len = sample_data.shape[0]
            data.append(sample_data)
            seq_lens.append(seq_len)
            label_index = self.word_to_index[sample['label']]
            labels[i - offset] = label_index
        max_seq_len = max(seq_lens)
        zero_padded_data = [np.append(s, np.zeros((max_seq_len - s.shape[0], s.shape[1])), axis=0) for s in data]
        data = np.stack(zero_padded_data)
        seq_lens = np.array(seq_lens)
        labels = np.array(labels)
        return {'x': data,
                'y': labels,
                'seq_len': seq_lens}


class SiameseSpeechCommandsDataCollector(SpeechCommandsDataCollector):
    """Data collector for Speech Commands dataset with extended functionality for training triplet networks
    """

    def get_duplicates(self, labels, offset, mode):
        """Gather samples of the same class as in the labels argument from the data set, applying transformations as needed.
        Output labels exactly match the input labels.

        Parameters
        ----------
        labels: list
            List of labels to sample
        offset: int
            Where to start when fetching deterministically.
        mode: str
            Which partition to use, must be 'training', 'validation', or
            'testing'.

        Returns
        -------
        dict
          List of sample data for the transformed samples, and list of label indexes, along with length of the data
        """

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
        """Gather samples from the data set, applying transformations as needed.
         As opposed to get_puplicates, this method makes sure that output data
         has different labels compared to the input labels.

        Parameters
        ----------
        labels: list
            List of labels
        offset: int
            Where to start when fetching deterministically.
        mode: str
            Which partition to use, must be 'training', 'validation', or
            'testing'.

        Returns
        -------
        dict
          List of sample data for the transformed samples, and list of label indexes, along with length of the data
        """
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