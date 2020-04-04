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

from scipy.io import wavfile as wav
from python_speech_features import mfcc, fbank, delta


class AudioPreprocessor(object):
    """Computes MFCC of an audio
    """

    def __init__(self, numcep=40, winlen=0.025, winstep=0.025, **kwargs):
        """Initializes processing parameters

        Parameters
        ----------
        numcep : int, optional
            Number of cepstral coefficients in the output (default is 40)
        winlen : float, optional
            Length of the window in seconds (default is 0.025)
        winstep: float, optional
            Stride of the window in seconds (default is 0.025)
        kwargs
        """

        self.__numcep = numcep
        self.__winlen = winlen
        self.__winstep = winstep
        return

    def __call__(self, *args, **kwargs):
        """Gets either input audio array ot audio filename and runs processing on the audio

        Parameters
        ----------
        args:
        kwargs

        Returns
        -------
        """

        if len(args) == 1 and isinstance(args[0], str):
            return self.__from_file(args[0])
        elif len(args) == 2 and isinstance(args[0], np.ndarray) and isinstance(args[1], int):
            return self.__from_array(args[0], args[1])
        else:
            raise TypeError("Expected either filename for audio file as an argument either raw audio with defined sample rate")

    def __from_array(self, input, sr):
        """Computes MFCC of the audio array

        Parameters
        ----------
        input: numpy.array
            1-dimensional numpy.array
        sr: int
            sampling rate of the audio signal

        Returns
        -------
        np.array
            2-dimensional array of MFCC
        """

        if len(input.shape) >= 2:
            inp_shape = input.shape
            raise ValueError(f"input shape has to be N*1, got: {inp_shape}")
        mfcc_out = mfcc(input, samplerate=sr, numcep=self.__numcep + 1, winlen=self.__winlen, nfft=int(sr*self.__winlen), winstep=self.__winstep)
        return mfcc_out[:, 1:]

    def __from_file(self, fname):
        """Loads audio from file and computes MFCC

        Parameters
        ----------
        fname: str
            filename of the audio file

        Returns
        -------
        np.array
            2-dimensional array of MFCC
        """

        rate, input = wav.read(fname)
        out = self.__from_array(input, sr=rate)
        return out

    def get_dim(self):
        """Returns output dimensionality, which is number of cepstral coefficients

        Returns
        -------
        int
            Returns number of cepstral coefficients
        """

        return self.__numcep


class AudioPreprocessorMFCCDeltaDelta(AudioPreprocessor):
    """Computes MFCC and it first and second derivative of an audio

    """

    def __init__(self, numcep=40, winlen=0.025, winstep=0.025, **kwargs):
        super(AudioPreprocessorMFCCDeltaDelta, self).__init__(numcep=numcep, winlen=winlen, winstep=winstep, **kwargs)
        return

    def __from_array(self, input, sr):
        """Computes MFCC and its first and second derivative of the audio array

        Parameters
        ----------
        input: numpy.array
            1-dimensional numpy.array
        sr: int
            sampling rate of the audio signal

        Returns
        -------
        np.array
            2-dimensional array of MFCC with its delta and delta delta
        """
        if len(input.shape) >= 2:
            inp_shape = input.shape
            raise ValueError(f"input shape has to be N*1, got: {inp_shape}")
        out = mfcc(input, samplerate=sr, numcep=self.__numcep + 1, winlen=self.__winlen, nfft=int(sr*self.__winlen), winstep=self.__winstep)[:, 1:]
        out_delta = delta(out, 1)
        out_delta_delta = delta(out_delta, 1)
        res = np.concatenate((out, out_delta, out_delta_delta), axis=1)
        return res

    def get_dim(self):
        """Returns output dimensionality, which is number of cepstral coefficients *3 (plus delta and delta delta)

        Returns
        -------
        int
            Returns number of output features
        """
        return self.__numcep * 3