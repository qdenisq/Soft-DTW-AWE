from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import math
import os.path
import random
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
# import tensorflow as tf

# # from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
# from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
from numpy.lib.stride_tricks import as_strided

from scipy.io import wavfile as wav
from python_speech_features import mfcc, fbank, delta

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

  Args:
    wanted_words: List of strings containing the custom words.

  Returns:
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

  Args:
    filename: File path of the data sample.
    validation_percentage: How much of the data set to use for validation.
    testing_percentage: How much of the data set to use for testing.

  Returns:
    String, one of 'training', 'validation', or 'testing'.
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

#
# class AudioProcessor(object):
#     def __init__(self,
#                  model_settings,
#                  data_dir,
#                  wanted_words,
#                  validation_percentage,
#                  testing_percentage,
#                  silence_percentage=0.,
#                  unknown_percentage=0.):
#         random.seed(RANDOM_SEED)
#         self.data_dir = data_dir
#         self.data_index = {"train": [],
#                            "test": [],
#                            "validation": []}
#         self._model_settings = model_settings
#         self.prepare_data_index(silence_percentage=silence_percentage,
#                                 unknown_percentage=unknown_percentage,
#                                 wanted_words=wanted_words,
#                                 validation_percentage=validation_percentage,
#                                 testing_percentage=testing_percentage)
#         self._build_processing_graph()
#         strip_window_size_ms = self._model_settings["strip_window_size_ms"]
#         strip_window_stride_ms = self._model_settings["strip_window_stride_ms"]
#         strip_window_size_samples = int(strip_window_size_ms / self._model_settings["window_stride_ms"])
#         strip_window_stride_samples = int(strip_window_stride_ms / self._model_settings["window_stride_ms"])
#         self._model_settings['strip_window_size_samples'] = strip_window_size_samples
#         self._model_settings['strip_window_stride_samples'] = strip_window_stride_samples
#         self._model_settings['strip_array_length'] = \
#          int(math.ceil((self._model_settings['spectrogram_length'] - strip_window_size_samples) / strip_window_stride_samples))
#         return
#
#     def prepare_data_index(self, silence_percentage, unknown_percentage,
#                            wanted_words, validation_percentage,
#                            testing_percentage):
#         """Prepares a list of the samples organized by set and label.
#
#         The training loop needs a list of all the available data, organized by
#         which partition it should belong to, and with ground truth labels attached.
#         This function analyzes the folders below the `data_dir`, figures out the
#         right
#         labels for each file based on the name of the subdirectory it belongs to,
#         and uses a stable hash to assign it to a data set partition.
#
#         Args:
#           silence_percentage: How much of the resulting data should be background.
#           unknown_percentage: How much should be audio outside the wanted classes.
#           wanted_words: Labels of the classes we want to be able to recognize.
#           validation_percentage: How much of the data set to use for validation.
#           testing_percentage: How much of the data set to use for testing.
#
#         Returns:
#           Dictionary containing a list of file information for each set partition,
#           and a lookup map for each class to determine its numeric index.
#
#         Raises:
#           Exception: If expected files are not found.
#         """
#         # Make sure the shuffling and picking of unknowns is deterministic.
#         random.seed(RANDOM_SEED)
#         wanted_words_index = {}
#         for index, wanted_word in enumerate(wanted_words):
#             wanted_words_index[wanted_word] = index + 2
#         self.data_index = {'validation': [], 'testing': [], 'training': []}
#         unknown_index = {'validation': [], 'testing': [], 'training': []}
#         all_words = {}
#         # Look through all the subfolders to find audio samples
#         search_path = os.path.join(self.data_dir, '*', '*.wav')
#         for wav_path in gfile.Glob(search_path):
#             _, word = os.path.split(os.path.dirname(wav_path))
#             word = word.lower()
#             # Treat the '_background_noise_' folder as a special case, since we expect
#             # it to contain long audio samples we mix in to improve training.
#             if word == BACKGROUND_NOISE_DIR_NAME:
#                 continue
#             all_words[word] = True
#             set_index = which_set(wav_path, validation_percentage, testing_percentage)
#             # If it's a known class, store its detail, otherwise add it to the list
#             # we'll use to train the unknown label.
#             if word in wanted_words_index:
#                 self.data_index[set_index].append({'label': word, 'file': wav_path})
#             else:
#                 unknown_index[set_index].append({'label': word, 'file': wav_path})
#         if not all_words:
#             raise Exception('No .wavs found at ' + search_path)
#         for index, wanted_word in enumerate(wanted_words):
#             if wanted_word not in all_words:
#                 raise Exception('Expected to find ' + wanted_word +
#                                 ' in labels but only found ' +
#                                 ', '.join(all_words.keys()))
#         # We need an arbitrary file to load as the input for the silence samples.
#         # It's multiplied by zero later, so the content doesn't matter.
#         silence_wav_path = self.data_index['training'][0]['file']
#         for set_index in ['validation', 'testing', 'training']:
#             set_size = len(self.data_index[set_index])
#             silence_size = int(math.ceil(set_size * silence_percentage / 100))
#             for _ in range(silence_size):
#                 self.data_index[set_index].append({
#                     'label': SILENCE_LABEL,
#                     'file': silence_wav_path
#                 })
#             # Pick some unknowns to add to each partition of the data set.
#             random.shuffle(unknown_index[set_index])
#             unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
#             self.data_index[set_index].extend(unknown_index[set_index][:unknown_size])
#         # Make sure the ordering is random.
#         for set_index in ['validation', 'testing', 'training']:
#             random.shuffle(self.data_index[set_index])
#         # Prepare the rest of the result data structure.
#         self.words_list = prepare_words_list(wanted_words)
#         self.word_to_index = {}
#         for word in all_words:
#             if word in wanted_words_index:
#                 self.word_to_index[word] = wanted_words_index[word]
#             else:
#                 self.word_to_index[word] = UNKNOWN_WORD_INDEX
#         self.word_to_index[SILENCE_LABEL] = SILENCE_INDEX
#
#     def set_size(self, mode):
#         """Calculates the number of samples in the dataset partition.
#
#         Args:
#           mode: Which partition, must be 'training', 'validation', or 'testing'.
#
#         Returns:
#           Number of samples in the partition.
#         """
#         return len(self.data_index[mode])
#
#     def _build_processing_graph(self):
#         """Builds a TensorFlow graph to apply the input distortions.
#
#             Creates a graph that loads a WAVE file, decodes it, scales the volume,
#             shifts it in time, adds in background noise, calculates a spectrogram, and
#             then builds an MFCC fingerprint from that.
#
#             This must be called with an active TensorFlow session running, and it
#             creates multiple placeholder inputs, and one output:
#
#               - wav_filename_placeholder_: Filename of the WAV to load.
#               - mfcc_: Output 2D fingerprint of processed audio.
#
#             """
#         with tf.name_scope('audio_processing'):
#             desired_samples = self._model_settings['desired_samples']
#             self.wav_filename_placeholder_ = tf.placeholder(tf.string, [])
#             wav_loader = io_ops.read_file(self.wav_filename_placeholder_)
#             wav_decoder = contrib_audio.decode_wav(
#                 wav_loader, desired_channels=1, desired_samples=desired_samples)
#             background_clamp = tf.clip_by_value(wav_decoder.audio, -1.0, 1.0)
#             # Run the spectrogram and MFCC ops to get a 2D 'fingerprint' of the audio.
#             spectrogram = contrib_audio.audio_spectrogram(
#                 background_clamp,
#                 window_size=self._model_settings['window_size_samples'],
#                 stride=self._model_settings['window_stride_samples'],
#                 magnitude_squared=True)
#             self.mfcc_ = contrib_audio.mfcc(
#                 spectrogram,
#                 wav_decoder.sample_rate,
#                 dct_coefficient_count=self._model_settings['dct_coefficient_count'])
#
#     def get_sequential_data_sample(self, fname):
#         sess = tf.get_default_session()
#         assert(sess is not None)
#         feed_dict = {
#             self.wav_filename_placeholder_: fname
#         }
#         mfcc = sess.run(self.mfcc_, feed_dict=feed_dict)
#
#         strip_window_size_samples = self._model_settings['strip_window_size_samples']
#         strip_window_stride_samples = self._model_settings['strip_window_stride_samples']
#         mfcc = np.squeeze(mfcc)
#         mfcc_chunks = np.asarray([mfcc[base:base + strip_window_size_samples]
#                                   for base in range(0, len(mfcc) - strip_window_size_samples, strip_window_stride_samples)])
#         return mfcc_chunks, mfcc
#
#     def get_data(self, how_many, offset, mode):
#         """Gather samples from the data set, applying transformations as needed.
#
#         When the mode is 'training', a random selection of samples will be returned,
#         otherwise the first N clips in the partition will be used. This ensures that
#         validation always uses the same samples, reducing noise in the metrics.
#
#         Args:
#           how_many: Desired number of samples to return. -1 means the entire
#             contents of this partition.
#           offset: Where to start when fetching deterministically.
#           mode: Which partition to use, must be 'training', 'validation', or
#             'testing'.
#
#         Returns:
#           List of sample data for the transformed samples, and list of label indexes
#         """
#         # Pick one of the partitions to choose samples from.
#         candidates = self.data_index[mode]
#         if how_many == -1:
#             sample_count = len(candidates)
#         else:
#             sample_count = max(0, min(how_many, len(candidates) - offset))
#         # Data and labels will be populated and returned.
#         num_chunks = self._model_settings['strip_array_length']
#         num_dct = self._model_settings['dct_coefficient_count']
#         window_size_samples = self._model_settings['strip_window_size_samples']
#         data = np.zeros((sample_count, num_chunks, window_size_samples, num_dct))
#         labels = np.zeros(sample_count)
#         desired_samples = self._model_settings['desired_samples']
#         pick_deterministically = (mode != 'training')
#         # Use the processing graph we created earlier to repeatedly to generate the
#         # final output sample data we'll use in training.
#         sess = tf.get_default_session()
#         assert(sess is not None)
#         for i in xrange(offset, offset + sample_count):
#             # Pick which audio sample to use.
#             if how_many == -1 or pick_deterministically:
#                 sample_index = i
#             else:
#                 sample_index = np.random.randint(len(candidates))
#             sample = candidates[sample_index]
#             # Run the graph to produce the output audio.
#             data[i - offset, :, :, :], _ = self.get_sequential_data_sample(sample['file'])
#             label_index = self.word_to_index[sample['label']]
#             labels[i - offset] = label_index
#         return data, labels


class SpeechCommandsDataCollector(object):
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

        Args:
          silence_percentage: How much of the resulting data should be background.
          unknown_percentage: How much should be audio outside the wanted classes.
          wanted_words: Labels of the classes we want to be able to recognize.
          validation_percentage: How much of the data set to use for validation.
          testing_percentage: How much of the data set to use for testing.

        Returns:
          Dictionary containing a list of file information for each set partition,
          and a lookup map for each class to determine its numeric index.

        Raises:
          Exception: If expected files are not found.
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

        Args:
          mode: Which partition, must be 'training', 'validation', or 'testing'.

        Returns:
          Number of samples in the partition.
        """
        return len(self.data_index[mode])

    def get_sequential_data_sample(self, fname):
        return self.__audio_preproc(fname)

    def get_data(self, how_many, offset, mode):
        """Gather samples from the data set, applying transformations as needed.

        When the mode is 'training', a random selection of samples will be returned,
        otherwise the first N clips in the partition will be used. This ensures that
        validation always uses the same samples, reducing noise in the metrics.

        Args:
          how_many: Desired number of samples to return. -1 means the entire
            contents of this partition.
          offset: Where to start when fetching deterministically.
          mode: Which partition to use, must be 'training', 'validation', or
            'testing'.

        Returns:
          List of sample data for the transformed samples, and list of label indexes
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


class AudioPreprocessor(object):
    def __init__(self, numcep=40, winlen=0.025, winstep=0.025, **kwargs):
        self.__numcep = numcep
        self.__winlen = winlen
        self.__winstep = winstep
        return

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], str):
            return self.__from_file(args[0])
        elif len(args) == 2 and isinstance(args[0], np.ndarray) and isinstance(args[1], int):
            return self.__from_array(args[0], args[1])
        else:
            raise TypeError("Expected either filename for audio file as an argument either raw audio with defined sample rate")

    def __from_array(self, input, sr):
        if len(input.shape) >= 2:
            inp_shape = input.shape
            raise ValueError(f"input shape has to be N*1, got: {inp_shape}")
        out = mfcc(input, samplerate=sr, numcep=self.__numcep + 1, winlen=self.__winlen, nfft=int(sr*self.__winlen), winstep=self.__winstep)
        return out[:, 1:]

    def __from_file(self, fname):
        rate, input = wav.read(fname)
        out = self.__from_array(input, sr=rate)
        return out

    def get_dim(self):
        return self.__numcep


class AudioPreprocessorMFCCDeltaDelta(object):
    def __init__(self, numcep=40, winlen=0.025, winstep=0.025, **kwargs):
        self.__numcep = numcep
        self.__winlen = winlen
        self.__winstep = winstep
        return

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], str):
            return self.__from_file(args[0])
        elif len(args) == 2 and isinstance(args[0], np.ndarray) and isinstance(args[1], int):
            return self.__from_array(args[0], args[1])
        else:
            raise TypeError("Expected either filename for audio file as an argument either raw audio with defined sample rate")

    def __from_array(self, input, sr):
        if len(input.shape) >= 2:
            inp_shape = input.shape
            raise ValueError(f"input shape has to be N*1, got: {inp_shape}")
        out = mfcc(input, samplerate=sr, numcep=self.__numcep + 1, winlen=self.__winlen, nfft=int(sr*self.__winlen), winstep=self.__winstep)[:, 1:]
        out_delta = delta(out, 1)
        out_delta_delta = delta(out_delta, 1)
        res = np.concatenate((out, out_delta, out_delta_delta), axis=1)
        return res

    def __from_file(self, fname):
        rate, input = wav.read(fname)
        out = self.__from_array(input, sr=rate)
        return out

    def get_dim(self):
        return self.__numcep * 3


class AudioPreprocessorFbank(object):
    def __init__(self, nfilt=40, winlen=0.025, winstep=0.025, **kwargs):
        self.__nfilt = nfilt
        self.__winlen = winlen
        self.__winstep = winstep
        return

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], str):
            return self.__from_file(args[0])
        elif len(args) == 2 and isinstance(args[0], np.ndarray) and isinstance(args[1], int):
            return self.__from_array(args[0], args[1])
        else:
            raise TypeError("Expected either filename for audio file as an argument either raw audio with defined sample rate")

    def __from_array(self, input, sr):
        if len(input.shape) >= 2:
            inp_shape = input.shape
            raise ValueError(f"input shape has to be N*1, got: {inp_shape}")
        feat, energy = fbank(input, sr, self.__winlen, self.__winstep, self.__nfilt, int(sr*self.__winlen), 0, sr//2, 0.97, lambda x: np.ones((x,)))
        # out = np.log10(feat)
        out = feat
        return out[:, :]

    def __from_file(self, fname):
        rate, input = wav.read(fname)
        input = input / (2 ** 15 - 1)
        out = self.__from_array(input, sr=rate)
        return out

    def get_dim(self):
        return self.__nfilt

