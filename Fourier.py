import numpy as np
import wave
import struct
from scipy.signal import butter,filtfilt

##### import contextlib


def find_peak_freq(f_transf):
    """ finds the maximum peaking frequency in a Fourier transform and returns
    the peak frequency index (Hz).
    Parameters:
    - f_transf (array_like): Fourier transform amplitudes (length of F-transform
    complex vectors) with frequency indexes.
    Limits: if 2 frequencies give the same amplitude, it will return the lowest
    frequency, max must be greater than 0.
    """
    peak_index = 0
    max_value = 0

    for freq, value in enumerate(f_transf):
        if value > max_value:
            max_value = value
            peak_index = freq

    return peak_index



######## needs fixing
def isolate_fourier(frequencies, min_amplitude = 0):
    """Premature function: takes a list of Fourier transform amplitudes with
    frequency indexes, returns a list of frequencies that are a local maxima.
    """
    index = 0
    maximums = [] ## list of maximum t.p. frequencies
    # make sure the frequencies are at least half the maximum, minimising noise
    min_amplitude = max(frequencies)/2

    for f in frequencies:
        if f > min_amplitude:
            # for each frequency, check the frequencies either side of it,
            # to see if it is a maximum
            if (frequencies[index + 1] <= f) and (frequencies[index - 1] <= f):
                maximums.append(index)
        index += 1
    return maximums



def pos_fft(data, frame_rate = 44100):
    """ From a block of data (audio samples) computes a Fourier transform and
    outputs the array of frequencies.
    Parameters:
    - data (array_like): audio samples data, sampled at frame_rate
    - frame_rate (int, optional): frame rate the audio was sampled at (samples).
    """

    ## Fourier transform for full sampled data
    data_fft = np.fft.fft(data, frame_rate, axis=0)

    ## Modulus of the complex number from Fourier, as a fn of freq
    frequencies = np.abs(data_fft)
    return frequencies



def limit_fft(fft_data, min_freq=0, max_freq=4400, incl=True):
    """Takes already transformed fft data and isolates to certain frequency
    range.
    Parameters:
    - fft_data (array_like): already transformed positive data,
        index=frequencies, element=magnitude of signal.
    - min_freq (int): minimum frequency needed (inclusive).
    - max_freq (int): maximum frequency needed (exclusive).
    - incl (bool): inclusive limits if True, exclusive limits if False.
    """
    if incl:
        lim_fft_data = fft_data[min_freq:max_freq + 1]
    else:
        lim_fft_data = fft_data[min_freq + 1:max_freq]
    return lim_fft_data



def get_fft(data, frame_rate = 44100, min_freq=0, max_freq=4400, incl=True):
    """From a block of data (audio samples) computes a Fourier transform,
    isolates the required frequencies and outputs the array of frequencies.
    Parameters:
    - data (array_like): audio samples data, sampled at frame_rate
    - frame_rate (int, optional): frame rate the audio was sampled at (samples).
    - min_freq (int): minimum frequency needed (inclusive).
    - max_freq (int): maximum frequency needed (exclusive).
    - incl (bool): inclusive limits if True, exclusive limits if False.
    """
    frequencies = pos_fft(data, frame_rate=frame_rate)
    lim_fft_data = limit_fft(frequencies, min_freq=min_freq, max_freq=max_freq,
        incl=incl)
    return lim_fft_data



def find_freq(data, frame_rate = 44100):
    """ From a block of data (audio samples) computes a Fourier transform and
    finds the peak frequency in the block.
    Parameters:
    - data (array_like): audio samples data, sampled at frame_rate
    - frame_rate (int, optional): frame rate the audio was sampled at (samples).
    """

    ## Fourier transform for full sampled data
    frequencies = pos_fft(data, frame_rate)

    ## eliminate higher impossible frequencies
    ## highest possible note on a piano is 4400 Hz
    frequencies = limit_fft(frequencies, max_freq=4400)

    ## find the peak frequency in the list
    peak_freq = find_peak_freq(frequencies)
    return peak_freq



def find_freq_in_wav(o_wav_file, start_samp, end_samp, frame_rate):
    """ Takes an open wav file as a wave object, and finds the peak frequency
    in the block of data obtained.
    Parameters:
    - o_wav_file (wave): already opened wave object by wave.open('file', 'r')
    - start_samp (int): starting sample position in the wav file (incl.)
    - end_samp (int): ending sample position (incl.)
    - frame_rate(int): sample rate of wav file (samples/s).
    """

    sample_len = end_samp - start_samp
    ## set starting position in iteration as start_samp
    o_wav_file.setpos(start_samp)
    ## read wave as a list of samples in hexadecimal
    wav_data = o_wav_file.readframes(sample_len)

    ## unpacks hexadecimal as list of amplitudes
    data = struct.unpack('{n}h'.format(n = sample_len), wav_data)

    ## find the peak frequency in the aduio section
    peak_freq = find_freq(data, frame_rate)
    return peak_freq



def if_amplitude_over(data_window, amplitude):
    """ For a window of data computes if any points cross an amplitude
    threshold. Both positive and negative.
    Parameters:
    - data_window (array_like): array of integers representing a sound wave
    - amplitude (int): amplitude the data needs to be higher than to register
    Output:
    - over (bool): if there is a data point above amplitude
    """

    maximum = max(data_window)
    minimum = min(data_window)
    # higher than the max or lower tham the min
    return (maximum > amplitude) or (minimum < -amplitude)



def change_orientation(data):
    """ Changes the orientation of the wires by returning the negative data.
    """
    return -data


def notch_filter_reshape(b_notch, a_notch, data):
    """ Filters a signal removing a component, using the notch filter
    coefficients provided, but reshaping the data into a 1-D array before
    and after the filtering.
    """
    data_reshape = data[:,0]
    data_notched = filtfilt(b_notch, a_notch, data_reshape)
    data_notched = np.reshape(data_notched, (len(data_notched),1))
    return data_notched


def butter_highpass_filter(data, cutoff, fs, order=2):
    """ Isolates data above a certain frequency, cutoff
    Parameters:
    - data (array_like): data that is being transformed to cut off lower frequencies
    - cutoff (int): desired cutoff frequency (Hz)
    - fs (int): sampling frequency
    - order (int): order of polynomial that approximates the wave, default=2
    """
    # nyquist frequency (half the samplerate)
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    # output the filtered signal
    y = filtfilt(b, a, data)
    return y


def butter_lowpass_filter(data, cutoff, fs, order=2):
    """ Isolates data below a certain frequency, cutoff
    Parameters:
    - data (array_like): data that is being transformed to cut off higher frequencies
    - cutoff (int): desired cutoff frequency (Hz)
    - fs (int): sampling frequency
    - order (int): order of polynomial that approximates the wave, default=2
    """
    # nyquist frequency (half the samplerate)
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # output the filtered signal
    y = filtfilt(b, a, data)
    return y


def butter_highpass_filter_reshape(data, cutoff, fs, order=2):
    """ Same as butter_highpass_filter but reshapes a 2-D array into the first
    column 1-D array before and after filtering.
    """
    data_reshape = data[:,0]
    data_filtered = butter_highpass_filter(data_reshape, cutoff, fs, order)
    data_filtered = np.reshape(data_filtered, (len(data_filtered),1))
    return data_filtered


def butter_lowpass_filter_reshape(data, cutoff, fs, order=2):
    """ Same as butter_lowpass_filter but reshapes a 2-D array into the first
    column 1-D array before and after filtering.
    """
    data_reshape = data[:,0]
    data_filtered = butter_lowpass_filter(data_reshape, cutoff, fs, order)
    data_filtered = np.reshape(data_filtered, (len(data_filtered),1))
    return data_filtered


def filters_reshape(data, b_notch, a_notch, max_freq, min_freq, fs, order=2):
    """ Puts signal through notch filter, and two butter filters, reshaping to
    the 1D array before and reshaping to 2D after.
    """
    data_reshape = data[:,0]
    data_notched = filtfilt(b_notch, a_notch, data_reshape)
    data_filtered = butter_highpass_filter(data_notched, min_freq, fs, order)
    data_filtered = butter_lowpass_filter(data_filtered, max_freq, fs, order)
    data_filtered = np.reshape(data_filtered, (len(data_filtered),1))
    return data_filtered
