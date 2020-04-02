def get_default_model_settings():
    settings = {'data_dir': '/tmp/speech_dataset/',
                'background_volume': 0.1,
                'background_frequency': 0.8,
                'silence_percentage': 10.0,
                'unknown_percentage': 10.0,
                'time_shift_ms': 100.0,
                'testing_percentage': 10,
                'validation_percentage': 10,
                'sample_rate': 16000,
                'clip_duration_ms': 1000,
                'window_size_ms': 30.0,
                'window_stride_ms': 10.0,
                'strip_window_size_ms': 60.0,
                'strip_window_stride_ms': 20.0,
                'dct_coefficient_count': 40,
                'how_many_training_steps': '15000,3000',
                'eval_step_interval': 400,
                'learning_rate': '0.001, 0.0001',
                'batch_size': 100,
                'summaries_dir': '/tmp/retrain_logs',
                'wanted_words': 'yes,no,up,down,left,right,on,off,stop,go,bed,bird,cat,dog,eight,five,four,happy,house,marvin,nine,one,seven',
                'train_dir': '/tmp/speech_commands_train',
                'save_step_interval': 100,
                'start_checkpoint': '',
                'hidden_reccurent_cells_count': 100
                }

    sample_rate = settings['sample_rate']
    desired_samples = int(sample_rate * settings['clip_duration_ms'] / 1000)
    window_size_samples = int(sample_rate * settings['window_size_ms'] / 1000)
    window_stride_samples = int(sample_rate * settings['window_stride_ms'] / 1000)
    length_minus_window = (desired_samples - window_size_samples)
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
    fingerprint_size = settings['dct_coefficient_count'] * spectrogram_length

    label_count = len(settings['wanted_words'].strip(',')) + 2
    additional_settings = {
        'desired_samples': desired_samples,
        'window_size_samples': window_size_samples,
        'window_stride_samples': window_stride_samples,
        'spectrogram_length': spectrogram_length,
        'fingerprint_size': fingerprint_size,
        'sample_rate': sample_rate,
        'label_count': label_count
    }

    model_settings = {**settings, **additional_settings}
    return model_settings