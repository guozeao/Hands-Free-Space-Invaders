#!/usr/bin/env python
# coding: utf-8

""" Keegan: Integrated streamer and game code, works as a prototype.
Needs fixing:

Simple classifier code

Tested: working!


"""

# In[14]:


####################
# RELEVANT IMPORTS # ==================================================================================================================
####################

# GAME IMPORTS
import pygame
import os
import time
import random

# DATA IMPORTS
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy
import pickle
import struct, math, random, os
import tsfeatures as tf
import pandas as pd
import numpy as np
from cache_decorator import Cache
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.datasets import make_classification
### TODO: for final file - move fns and variables into this code
from SimpleClassifier import *
#from catch22 import catch22_all

# PHYSICS IMPORTS
import queue
import sys
#import time ## in game imports
import serial
import serial.tools.list_ports as ListPorts
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
#import numpy as np ## in data imports
import sounddevice as sd
from scipy import signal
### TODO: for final file - move fns and variables into this code
from Fourier import *
from serial.serialutil import SerialException


#########################
# GLOBALS AND CONSTANTS # ==============================================================================================================
#########################

CALIBRATION_NUMBER = 5
simple_orientation = 0


# GAME VARIABLES

# window size
WIDTH, HEIGHT = 750, 750

# Load images
RED_SPACE_SHIP = pygame.image.load(os.path.join("assets", "pixel_ship_red_small.png"))
GREEN_SPACE_SHIP = pygame.image.load(os.path.join("assets", "pixel_ship_green_small.png"))
BLUE_SPACE_SHIP = pygame.image.load(os.path.join("assets", "pixel_ship_blue_small.png"))

# Player player
YELLOW_SPACE_SHIP = pygame.image.load(os.path.join("assets", "pixel_ship_yellow.png"))

# Lasers
RED_LASER = pygame.image.load(os.path.join("assets", "pixel_laser_red.png"))
GREEN_LASER = pygame.image.load(os.path.join("assets", "pixel_laser_green.png"))
BLUE_LASER = pygame.image.load(os.path.join("assets", "pixel_laser_blue.png"))
YELLOW_LASER = pygame.image.load(os.path.join("assets", "pixel_laser_yellow.png"))


# DATA VARIABLES
wave_path = "Phys_Lab_31-3_BYB_Data/BYB_Recording_2021-03-31_14.36.25.wav"
training_path = "training_data"

statistics = {"flat_spots" : tf.flat_spots,
              "crossing_points" : tf.crossing_points,
              # UNCOMMENT THIS
              "acf_features" : tf.acf_features,
              "std": np.std} # TODO : Leo - which features are useful?

stat_functions = list(statistics.values())


# PHYSICS VARIABLES

## Audio variables:
# show the list of audio devices and exit
list_devices = False
# input device (numeric ID or substring)
audio_device = 1
# visible time slot (default: 200 ms)
audio_window = 200
# minimum time between plot updates (default: 30 ms)
audio_interval = 30
# audio block size (in samples)
audio_blocksize = 2205
# sampling rate of audio device
audio_samplerate = 44100
# display every Nth sample: (default: 10)
audio_downsample = 10
# input channels to plot (default: the first)
audio_channels = [1]
# microphone audio threshold for shooting
mic_threshold = 0.1

# channels are for more than 1 input, e.g. L/R, 2 mics.
if any(c < 1 for c in audio_channels):
    print("error: argument CHANNEL: must be >= 1")
    exit()
mapping = [c - 1 for c in audio_channels]  # Channel numbers start with 1


# q_1 is audio queue
q_1 = queue.Queue()
# q_2 is spiker queue
q_2 = queue.Queue()

# Spikerbox variables:
baudrate = 1000
cport = 'COM8'  # set the correct port before you run it
#cport = '/dev/tty.usbmodem141101'  # set the correct port before run it

inputBufferSize = 2000 # keep betweein 2000-20000

spiker_samplerate = 9988 ###### look at this

spiker_downsample = 1

total_time = 20.0; # time in seconds [[1 s = 20000 buffer size]]
max_time = 0.5; # time plotted in window [s]
N_loops = 20000.0/inputBufferSize*total_time

T_acquire = inputBufferSize/20000.0    # length of time that data is acquired for
N_max_loops = max_time/T_acquire    # total number of loops to cover desire time window

shift_length = 0



# initialise the global window variables
plotdata1 = np.zeros((1, 1))
plotdata2 = np.zeros((1, 1))


##################
# DATA FUNCTIONS # =====================================================================================================
##################




def read_wave(path):
    if not os.path.exists(path):
        print(f"FILE NOT FOUND: {path}")
        exit()
    else:
        samplerate, raw_data = wavfile.read(path)
        time = np.array(range(0,len(raw_data)))/samplerate
        wave = { 'y' : raw_data, 'time' : time, 'samplerate' : samplerate }
        return wave

def generate_statistics(wave, predict=False):
    """
    Generates the globally defined statistics* about a given wave (* see variable 'statistics')

    @param wave        : wave object
    @type wave         : dict
    @return colnames   : names of generated statistics (note: has wave_stats ordering)
    @rtype colnames    : list
    @return wave_stats : generated statistics (note: has colnames ordering)
    @rtype wave_stats  : list
    """
    stats = {}
    # Keegan: array is 2D, so take the 1st column vector of the array as the wave.
    if predict:
        Y = wave['y'][:,0]
    else:
        Y = wave['y'][:,0]
    
    for fun in stat_functions:
        dic = fun(Y)
        try:
            for k,v in dic.items():
                stats[k] = v
        except:

            stats["std"] = dic

    colnames = list(stats.keys())
    wave_stats = list(stats.values())
    return colnames, wave_stats

def plot_wave(wave):
    plt.plot(wave['y'])
    plt.show()

def balance_dataset(data):
    len_r = len(data[data["labels"]=="1"])
    len_l = len(data[data["labels"]=="0"])

    smallest = len_l if len_r > len_l else len_r

    data_right = data[data["labels"]=="1"].sample(smallest, random_state=0)
    data_left = data[data["labels"]=="0"].sample(smallest, random_state=0)
    dat = pd.concat([data_right, data_left])
    return dat


### this function does not yet work for multinomial but we can just use the binary result
### selected features (also with manual feature selection) are given in the build_classifier function
def lasso_feature_selection(training_path):
    dataset = read_training_data(training_path)

    print(dataset)

    X = dataset.drop(["labels"], axis=1).values
    y = dataset["labels"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    # Scale the data
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    # for feature selection
    column_names = list(dataset.drop(["labels"], axis=1).columns)

    # Define the multinomial logistic regression model
    #solver='lbfgs'

    classifier = LogisticRegression(solver='saga', penalty="l1")
    classifier.fit(X_train, y_train)
    coefficients = classifier.coef_[0]
    #print(coefficients)

    selected_features = []
    for i,column in enumerate(column_names):
        if coefficients[i] != 0:
            selected_features.append(column)

    '''coef_dict = {}
    for coef, feat in zip(classifier.coef_[0,:],column_names):
        coef_dict[feat] = coef'''

    #print("Selected features: "+str(selected_features))
    #print("All: "+str(column_names))

def build_classifier(training_path, classifier = "LR", show_acc = False):
    # REFERENCE: https://in.springboard.com/blog/logistic-regression-explained/
    """
    Builds the specific classifier - default is Logistic Regression - and trains it on the training data.

    @param training_path : local directory containining training data, i.e., .wav files and .txt files
    @type training_path  : str
    @param classifier    : which classifier to use
    @type classifier     : str
    @param show_acc      : flag used to show classifier accuracy
    @type show_acc       : boolean
    @return classifier   : linear regression classifier
    @rtype               : sklearn.linear_model type object
    """
    dataset = read_training_data(training_path)

    ########## left and right only ################
    #dataset = dataset[dataset["labels"]!="2"]

    ######## specify the features after running lasso feature selection ##########
    #dataset = dataset[['diff2_acf10', 'stdev', 'labels']]
    dataset = dataset[['flat_spots', 'crossing_points', 'diff2_acf10', 'std', 'labels']]

    # Take necessary rows and split labels into new dataset, y
    X = dataset.drop(["labels"], axis=1).values
    y = list(dataset["labels"])
    # Divide into training (75%) and testing (25%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    # Scale the data
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    ####### use this to test on the whole dataset #######
    #sc_X = StandardScaler()
    #X = sc_X.fit_transform(X)

    ##### multilinear ######
    #classifier = LogisticRegression(multi_class='multinomial', solver='saga')

    ######## binary classifier, l2 reg ##########
    classifier = LogisticRegression(multi_class= "multinomial", solver='saga', penalty = 'l2')
    #########################
    classifier.fit(X_train, y_train)
    if show_acc:
        determine_accuracy(classifier, X_test, y_test)

    # predict the class label
    # row = generate_statistics(wave)
    # y_hat = model.predict([row])
    # OR predict a multinomial probability distribution
    # y_hat = model.predict_proba([row])
    #classifier.predict(X_test)
    return classifier, sc_X

def predict(wave, classifier, standard_scaler):
    colnames, wave_stats = generate_statistics(wave, predict=True)
    # get rid of nan values
    #wave_stats[np.isnan(wave_stats)] = 0
    wave_stats_reshape = np.reshape(wave_stats, (1,-1))
    # create a dataframe
    wave_stats_dataframe = pd.DataFrame(wave_stats_reshape, columns = colnames)
    # select specific stats
    wave_stats = wave_stats_dataframe[['flat_spots', 'crossing_points', 'diff2_acf10', 'std']]
    # just the values in an array
    wave_stats = np.array(wave_stats.values)
    # normalise

    wave_stats = standard_scaler.transform(wave_stats)
    # predict.
    label = classifier.predict(wave_stats)
    return label


def determine_accuracy(model, X, y):
    """
    Runs repeated CV on the model and returns the mean and sd of the accuracy to 3 decimal places.

    @param model : Fitted logisitic regression model
    @type model  : sklearn.linear_model type object
    @param X     : All data (minus labels)
    @type X      : numpy array
    @param y     : Corresponding labels
    @type y      : numpy array
    """
    # make fake dataset (can delete later)
    # X, y = make_classification(n_samples=72, n_features=5, n_informative=5, n_redundant=0, n_classes=3, random_state=1)
    y_true = list(y)
    y_pred = model.predict(X)

    print(classification_report(y_true, y_pred))

    # define the model evaluation procedure
    # cv = RepeatedStratifiedKFold(n_splits=1, n_repeats=3, random_state=1)
    # evaluate the model and collect the scores
    # n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # report the model performance

    # print('Mean Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

def convert_format(raw_data, samplerate):
    time = np.array(range(0,len(raw_data)))/samplerate
    wave = { 'y' : raw_data, 'time' : time, 'samplerate' : samplerate }
    return wave


def find_eye_movements(wave,
                        window_size,
                        name_str = None,
                        threshold = {'zc' : 600},
                        ds_rate = 50):
    """
    Downsamples the given wave and uses hot frame technique to determine any eye movement intervals.

    @param wave        : raw wave values (y) and its samplerate (samplerate)
    @type wave         : dictionary
    @param window_size : size of the
    @type window_size  : float
    @param threshold   : statistic used to identify movement and corresponding threshold
    @type threshold    : dictionary like { 'statistic' : threshold }
    @param ds_rate     : downsampling rate
    @type ds_rate      : int
    """

    count_f = 0
    lower_interval = 0
    time = wave['time']
    Y = wave['y']
    samplerate = wave['samplerate']
    max_time = len(time)
    i = 0
    while (lower_interval + window_size < max_time):
        upper_interval = lower_interval + window_size
        i+=1
        wave_slice = Y[lower_interval:upper_interval] # change indexing to time
        # check for zero crossings
        #zc, zc_idx = zero_crossings(wave_slice)
        stdev = np.std(wave_slice)

        if (stdev >= 80): #zc >= thresholdEvents):
            increment = int(window_size/2)
            upper_temp = upper_interval
            lower_temp = lower_interval
            start = lower_interval
            end = upper_interval

            #### find end of signal ###
            while (True):

                #zc_end,_ = zero_crossings(wave_slice[upper_temp : int(upper_temp + increment)])
                ws = Y[int(upper_temp) : int(upper_temp + increment)]

                stdev = np.std(Y[int(upper_temp) : int(upper_temp + increment)])
                if (stdev <= 90):#(zc_end < thresholdEvents):
                    # found end
                    break
                end = upper_temp + increment
                upper_temp = upper_temp + increment

            #### find start of signal ###
            increment_st = int(window_size/2)
            while (True):

                #zc_end,_ = zero_crossings(wave_slice[upper_temp : int(upper_temp + increment)])
                ws = Y[int(lower_temp) : int(lower_temp + increment_st)]

                stdev = np.std(Y[int(lower_temp) : int(lower_temp + increment_st)])
                if (stdev >= 90):#(zc_end < thresholdEvents):
                    # found end
                    break
                start = lower_temp + increment_st
                lower_temp = lower_temp + increment_st
            # save slice
            # if ((hotframe_end == -1 ) or (i >= hotframe_end)):

            signal_slice = Y[start:end]
            time_slice = time[start:end]
            lower_interval = end

            # plt.scatter(time_slice,signal_slice)
            # plt.show()
            filename = str(i)+"_"+name_str
            scipy.io.wavfile.write(filename, samplerate, wave_slice)
            count_f += 1
        else:
            lower_interval =lower_interval + window_size

def split_data(path):
    """
    Splits each .wav file into separate files, one per each signal

    @path : string path to .wav file
    @type seq  : numpy array
    @return zc : number of zero crossings
    @rtype     : int
    """
    wave = read_wave(path)
    time = np.array(range(0,len(wave['y'])))/wave["samplerate"]

    time_seconds = np.array(range(0,len(wave['y'])))/wave["samplerate"]

    plt.plot(time_seconds, wave['y'])
    plt.show()

    find_eye_movements(wave['y'], time, wave["samplerate"],windowSize = wave["samplerate"], name_str="16.45.43")


def determine_movement(wave, ds_rate = 50, method = {"ZC" : 20}):
    """
    For any given wave sequence determines if there is an eye movement using a specified method.
    Used exclusively in conjunction with find_eye_movements.

    @param wave       : raw wave values (y) and its samplerate (samplerate)
    @type wave        : dictionary
    @param ds_rate     : downsampling rate
    @type ds_rate      : int
    @param method     : method - and corresponding threshold - used to determine eye movement
    @type method      : dictionary
    @return movement  : detection of a movement and its corresponding start and end time
    @trype            : dictionary like { movement : True, start_int : 0.5, end_int : 2 }
    """
    # Y = wave['y']
    # time = wave['time']
    # samplerate = wave['samplerate']
    # ds_idxs = [x for x in range(len(Y)) if x % ds_rate == 0]
    # for i, ds_point in enumerate(ds_idxs):

def zero_crossings(seq):
    """
    Calculates the number of zero crossings in any wave sequence

    @param seq : numpy array of downsampled wave values
    @type seq  : numpy array
    @return zc : number of zero crossings
    @rtype     : int
    """
    #print(seq)
    return (((seq[:-1] * seq[1:]) < 0).sum(), ((seq[:-1] * seq[1:]) < 0))

def smooth_wave(df):
    pass

@Cache()
def read_training_data(path, balance_data = False):
    """
    Reads in the training data .wav files from the specified local directory,
    generates statistics on each wave, and outputs the dataset.

    @param path     : path to local directory containing training data
    @type path      : str
    @return dataset : dataset like { waveId : [0, ...], stat1 : [s11, s12, ...], stat2 : [s21, s22, ...], ... }
    @rtype          : pandas DataFrame
    """
    data = {}
    labels = []
    if not os.path.exists(path):
        print("ERROR: PATH NOT VALID")
        return None
    for root, dirs, files in os.walk(path, topdown=False):
        # Iterating through the directory containing training data
        for name in files:
            if name.endswith(".wav"):

                wave_file = path + "/" + name
                # read the wave file
                wave = read_wave(wave_file)
                #print(wave)
                # TODO: Either use find_eye_movements function here to get intervals of movement for each training
                #       data .wav file OR we can use the txt files (still unsure how to interpret them though)

                ###### extract label from the filename ##########
                name_lst = wave_file.split("_")
                label = name_lst[len(name_lst)-1]
                if (label == "B.wav"):
                    label = "2" # blink

                else:
                    label = label.strip("-S.wav")
                    if (label =="R"):
                        label = "1"
                    else:
                        label = "0"

                labels.append(label)
                colnames, wave_stats = generate_statistics(wave)

                # Creating a dictionary in preparation for transformation to a dataframe
                for i, col in enumerate(colnames):
                    if not data.get(col):
                        # For the first iteration of the loop
                        data[col] = [wave_stats[i]]
                    else:
                        data[col].append(wave_stats[i])

            '''if name.endswith(".txt"):
                # TODO: Extract labels from txt files? i.e., left or right [or none?]
                labels_file = path + "/" + name
                labels = open(labels_file).read().split("\n")'''

    data["labels"] = labels
    ############ remove blinks for now ###############
    data = pd.DataFrame(data)

    #data = data[data["labels"]!="2"]
    if (balance_data):
        data = balance_dataset(data)
    return data

#####################
# PHYSICS FUNCTIONS # ===============================================================================================
#####################

def calibration_update_spiker_plotdata():
    """This is called to update and roll the plotdata2 array.
    Data is collected from the queue, downsampled and added to the end of the
    plotdata1 array.

    Typically, spikerbox callbacks happen more frequently than plotdata updates,
    therefore the queue tends to contain multiple blocks of spikerbox data.

    """

    global plotdata2
    global spiker_full_data
    global spiker_filt_data
    while True:
        try:
            data = q_2.get_nowait()
        except queue.Empty:
            break
        # add data to spiker_full_data for recording
        spiker_full_data = np.concatenate((spiker_full_data, data))

        # 50 Hz notch filter
        #data_notched = notch_filter_reshape(b_notch, a_notch, data)
        # filter frequencies outside of min-max frequency.
        #data_filtered = butter_lowpass_filter_reshape(data_notched,
            #spiker_max_freq, spiker_samplerate)
        #data_filtered = butter_highpass_filter_reshape(data_filtered,
            #spiker_min_freq, spiker_samplerate)

        # add data to spiker_filt_data for recording
        #spiker_filt_data = np.concatenate((spiker_filt_data, data_filtered))

        ## Stagnated data for easing graph input
        reduced_data = data[::audio_downsample, mapping]

        shift = len(reduced_data)
        plotdata2 = np.roll(plotdata2, -shift, axis=0)
        plotdata2[-shift:, :] = reduced_data

    return


def calibration_update_mic_plotdata():
    """This is called to update and roll the global plotdata1 array.
    Data is collected from the queue, downsampled and added to the end of the
    plotdata1 array.

    Typically, audio callbacks happen more frequently than plot updates,
    therefore the queue tends to contain multiple blocks of audio data.

    """
    global plotdata1
    global audio_full_data
    while True:
        try:
            data = q_1.get_nowait()
            # add data to audio_full_data for recording
            audio_full_data = np.concatenate((audio_full_data, data))
            ## Stagnated data for easing graph input
            reduced_data = data[::audio_downsample, mapping]
        except queue.Empty:
            break
        shift = len(reduced_data)
        plotdata1 = np.roll(plotdata1, -shift, axis=0)
        plotdata1[-shift:, :] = reduced_data

    return



# General physics functions
def block_fft_print(data, frame_rate):
    """Takes the audio data in a block,
    takes the fft of that audio block and translates that to its peak freq
    (which is prints).

    """
    peak_freq = find_freq(data, frame_rate)
    print(peak_freq)

def print_mean(data):
    """ Computes the mean of a data set (array_like) and prints.
    """
    m = np.mean(data)
    print(m)

def print_mean(data, type):
    """ Computes the mean of a data set (array_like) and prints.
    type is an int (1 or 2) which tells the function which dataset it is looking
    at (1 for audio data, 2 for spiker data) outputs accordingly.
    """

    m = np.mean(data)
    if type == 1:
        print("Mic mean is: {}".format(m))
    elif type == 2:
        print("Spiker mean is: {}".format(m))
    else:
        return

# Microphone functions
def audio_callback(indata, frames, time, status):
    """This is called by the InputStream (from a separate thread) for each audio
    block.
    Override function called in the Stream.
    This runs continuously in the InputStream placing audio data in the queue."""
    if status:
        print(status, file=sys.stderr)
    # Fancy indexing with mapping creates a (necessary!) copy:
    q_1.put(indata[:, mapping]) ## Full data here

def update_mic_plotdata():
    """This is called to update and roll the global plotdata1 array.
    Data is collected from the queue, downsampled and added to the end of the
    plotdata1 array.

    Typically, audio callbacks happen more frequently than plot updates,
    therefore the queue tends to contain multiple blocks of audio data.

    """
    global plotdata1
    while True:
        try:
            data = q_1.get_nowait()
            ## Stagnated data for easing graph input
            reduced_data = data[::audio_downsample, mapping]
        except queue.Empty:
            break
        shift = len(reduced_data)
        plotdata1 = np.roll(plotdata1, -shift, axis=0)
        plotdata1[-shift:, :] = reduced_data

    return

def print_mic_mean_loop():
    """Called by the mic_thread, to continuously update the plotdata1 array,
    and then print the mean of that array.
    Calls the print_mean() function and prints the mean of the global
    mic data. """
    global plotdata1
    while True:
        update_mic_plotdata()
        print_mean(plotdata1, 1)

def print_mic_mean():
    """Called by the mic_thread, to update the plotdata1 array,
    and then print the mean of that array.
    Calls the print_mean() function and prints the mean of the global
    mic data. """
    global plotdata1
    update_mic_plotdata()
    #print(plotdata1)
    print_mean(plotdata1, 1)

def classify_mic():
    """Reads the microphone window (plotdata1) and outputs a bool determining
    whether to shoot.
    """
    global plotdata1
    global mic_threshold
    # currently we will shoot if the audio amplitude signal is over 300.
    ### test and change.
    #print(max(plotdata1))
    return if_amplitude_over(plotdata1, mic_threshold)

# Spikerbox functions
def read_arduino(ser,inputBufferSize):
    """ Reads in the data from the spikerbox. Ouputs an array of integers."""
#    data = ser.readline(inputBufferSize)
    data = ser.read(inputBufferSize)
    # reads in bytes size,
    ## perhaps change to read_until()?
    out =[(int(data[i])) for i in range(0,len(data))]
    return out

def process_data(data):
    """ Changes the spikerbox data into a managable form. """
    data_in = np.array(data)
    result = []
    i = 1
    while i < len(data_in)-1:
        if data_in[i] > 127:
            # Found beginning of frame
            # Extract one sample from 2 bytes
            intout = (np.bitwise_and(data_in[i],127))*128
            i = i + 1
            intout = intout + data_in[i]
            result = np.append(result,intout)
        i=i+1
    return result

def read_arduino_and_process(ser, inputBufferSize):
    """Reads the spiker box arduino data with a set inputBufferSize
    #####from whatever is waiting in the buffer.
    Also processes the data using process_data(), discarding some data points.
    Outputs the processed data.

    """
    data = read_arduino(ser,inputBufferSize)
    data_processed = process_data(data) ##### this is taking 0.3 seconds
    # return processed data (centred around 512)
    return data_processed

def spiker_callback(ser, inputBufferSize):
    """Collects new data, and places in the queue."""
    data_processed = read_arduino_and_process(ser, inputBufferSize) ##### this is taking 0.3 seconds
    # reshape data into right size
    indata = np.reshape(data_processed, (len(data_processed),
        len(audio_channels)))
    # Fancy indexing with mapping creates a (necessary!) copy:
    q_2.put(indata[:, mapping]) ## Full data here

def spiker_callback_loop(ser, inputBufferSize):
    """Continuously calls the spiker_callback function, which collects data and
    places it in the queue. """

    while True:
        spiker_callback(ser, inputBufferSize)

def port_exists(cport):
    """ Finds if the port cport currently exists in the list of ports. """

    # current list of ports
    ports = ListPorts.comports()
    output = False
    # if any port matches cport, return true
    for port, desc, hwid in sorted(ports):
        if cport == port:
            output = True
    return output

def wrong_port(cport):
    """ Prints the output for the incorrect port selection error.
    Gives command line the option to change to correct port before exiting.
    Ouputs the correct new port (str).
    """

    print("\nError: could not open port '{}'".format(cport))
    print("\nAvailable ports are: ")
    # print each of the available ports
    ports = ListPorts.comports()
    for port, desc, hwid in sorted(ports):
        print("{}: {} [{}]".format(port, desc, hwid))

    # attempt to get a new port number
    new_port = input("\nWhich number port would you like to change to? ")
    new_cport = "COM" + new_port

    # check if new port is valid
    if port_exists(new_cport):
        return new_cport
    else:
        return wrong_port(new_cport)

def update_spiker_plotdata():
    """This is called to update and roll the plotdata2 array.
    Data is collected from the queue, downsampled and added to the end of the
    plotdata2 array.

    Typically, spikerbox callbacks happen more frequently than plotdata updates,
    therefore the queue tends to contain multiple blocks of spikerbox data.

    """

    global plotdata2
    shift = 0
    while True:
        try:
            data = q_2.get_nowait()
            if len(data) == 0:
                continue
            ## Stagnated data for easing graph input
            reduced_data = data[::spiker_downsample, mapping]
        except queue.Empty:
            break
        shift = len(reduced_data)
        plotdata2 = np.roll(plotdata2, -shift, axis=0)
        plotdata2[-shift:, :] = reduced_data

    return shift

def print_spiker_mean_loop():
    """ Continuously calls the print_mean() function and prints the mean of the
    global spikerbox data. """
    global plotdata2
    while True:
        update_spiker_plotdata()
        print_mean(plotdata2, 2)

def print_spiker_mean():
    """ Calls the print_mean() function and prints the mean of the global
    spikerbox data. """
    global plotdata2
    update_spiker_plotdata()
    print_mean(plotdata2, 2)

#####################
# GAME CLASSES # ===============================================================================================
#####################

class Laser:
    def __init__(self, x, y, img):
        self.x = x
        self.y = y
        self.img = img
        self.mask = pygame.mask.from_surface(self.img)

    def draw(self, window):
        window.blit(self.img, (self.x, self.y))

    def move(self, vel):
        self.y += vel

    def off_screen(self, height):
        return not(self.y <= height and self.y >= 0)

    def collision(self, obj):
        return collide(self, obj)


class Ship:
    COOLDOWN = 15

    def __init__(self, x, y, health=100):
        self.x = x
        self.y = y
        self.health = health
        self.ship_img = None
        self.laser_img = None
        self.lasers = []
        self.cool_down_counter = 0

    def draw(self, window):
        window.blit(self.ship_img, (self.x, self.y))
        for laser in self.lasers:
            laser.draw(window)

    def move_lasers(self, vel, obj):
        self.cooldown()
        for laser in self.lasers:
            laser.move(vel)
            if laser.off_screen(HEIGHT):
                self.lasers.remove(laser)
            elif laser.collision(obj):
                obj.health -= 10
                self.lasers.remove(laser)

    def cooldown(self):
        if self.cool_down_counter >= self.COOLDOWN:
            self.cool_down_counter = 0
        elif self.cool_down_counter > 0:
            self.cool_down_counter += 1

    def shoot(self):
        if self.cool_down_counter == 0:
            laser = Laser(self.x, self.y, self.laser_img)
            self.lasers.append(laser)
            self.cool_down_counter = 1

    def get_width(self):
        return self.ship_img.get_width()

    def get_height(self):
        return self.ship_img.get_height()


class Player(Ship):
    def __init__(self, x, y, health=100):
        super().__init__(x, y, health)
        self.ship_img = YELLOW_SPACE_SHIP
        self.laser_img = YELLOW_LASER
        self.mask = pygame.mask.from_surface(self.ship_img)
        self.max_health = health

    def move_lasers(self, vel, objs):
        self.cooldown()
        for laser in self.lasers:
            laser.move(vel)
            if laser.off_screen(HEIGHT):
                self.lasers.remove(laser)
            else:
                for obj in objs:
                    if laser.collision(obj):
                        objs.remove(obj)
                        if laser in self.lasers:
                            self.lasers.remove(laser)

    def draw(self, window):
        super().draw(window)
        self.healthbar(window)

    def healthbar(self, window):
        pygame.draw.rect(window, (255,0,0), (self.x, self.y + self.ship_img.get_height() + 10, self.ship_img.get_width(), 10))
        pygame.draw.rect(window, (0,255,0), (self.x, self.y + self.ship_img.get_height() + 10, self.ship_img.get_width() * (self.health/self.max_health), 10))


class Enemy(Ship):
    COLOR_MAP = {
                "red": (RED_SPACE_SHIP, RED_LASER),
                "green": (GREEN_SPACE_SHIP, GREEN_LASER),
                "blue": (BLUE_SPACE_SHIP, BLUE_LASER)
                }

    def __init__(self, x, y, color, health=100):
        super().__init__(x, y, health)
        self.ship_img, self.laser_img = self.COLOR_MAP[color]
        self.mask = pygame.mask.from_surface(self.ship_img)

    def move(self, vel):
        self.y += vel

    def shoot(self):
        if self.cool_down_counter == 0:
            laser = Laser(self.x-20, self.y, self.laser_img)
            self.lasers.append(laser)
            self.cool_down_counter = 1

#####################
# GAME FUNCTIONS # ===============================================================================================
#####################

def collide(obj1, obj2):
    offset_x = obj2.x - obj1.x
    offset_y = obj2.y - obj1.y
    return obj1.mask.overlap(obj2.mask, (offset_x, offset_y)) != None



###############
# CALIBRATION # =====================================================================================================
###############

def welcome_menu(show_banner = False):
    banner = """
██╗    ██╗███████╗██╗      ██████╗ ██████╗ ███╗   ███╗███████╗
██║    ██║██╔════╝██║     ██╔════╝██╔═══██╗████╗ ████║██╔════╝
██║ █╗ ██║█████╗  ██║     ██║     ██║   ██║██╔████╔██║█████╗
██║███╗██║██╔══╝  ██║     ██║     ██║   ██║██║╚██╔╝██║██╔══╝
╚███╔███╔╝███████╗███████╗╚██████╗╚██████╔╝██║ ╚═╝ ██║███████╗
 ╚══╝╚══╝ ╚══════╝╚══════╝ ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝
    """
    if show_banner:
        print(banner)

    # open spikerbox object and call it global variable ser
    global ser
    global cport
    # test to see if the port is open
    if not port_exists(cport):
        cport = wrong_port(cport)
    ser = serial.Serial(port=cport, baudrate=baudrate)

    options = [
        "Calibrate",
        "Play Game",
        "Quit"
    ]
    
    for i,e in enumerate(options):
        print(f"    [{i}] - {e}")

    # if game has been calibrated
    calibrated = False

    while True:
        print()
        choice = input("Please enter your decision.\n ~ ").split(" ")[0].capitalize()
        if choice == options[0] or choice == "0":
            # START CALIBRATION
            global mic_threshold
            global simple_orientation
            mic_threshold, simple_orientation = begin_calibration(show_banner)
            calibrated = True
            for i,e in enumerate(options):
                print(f"    [{i}] - {e}")

        elif choice == options[1] or choice == "1":
            if not calibrated:
                proceed = input("WARNING: You have no calibrated yet, are you sure you want to proceed? (y/n) ").lower()
                if proceed == "y" or proceed == "":
                    game_menu()
                    return
            else:
                game_menu()
                return
        elif choice == options[2] or choice == "2":
            print("\n\nThank you for using our program!")
            exit(1)

def begin_calibration(banner = False):
    banner = """
 ██████╗ █████╗ ██╗     ██╗██████╗ ██████╗  █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
██╔════╝██╔══██╗██║     ██║██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
██║     ███████║██║     ██║██████╔╝██████╔╝███████║   ██║   ██║██║   ██║██╔██╗ ██║
██║     ██╔══██║██║     ██║██╔══██╗██╔══██╗██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
╚██████╗██║  ██║███████╗██║██████╔╝██║  ██║██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
 ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
"""
    if banner:
        print(banner)

    # length of recording in seconds
    t_stop = 2

    global audio_full_data
    global spiker_full_data

    # global audio full data being recorded:
    audio_full_data = np.zeros((0, len(audio_channels)))
    # set as an array of integers
    audio_full_data = audio_full_data.astype('int32')

    # global raw spiker full data being recorded:
    spiker_full_data = np.zeros((0, len(audio_channels)))
    # set as an array of integers
    spiker_full_data = spiker_full_data.astype('int32')

    global ser

    # take continuous data stream
    ser.timeout = inputBufferSize/20000.0  # set read timeout, 20000 is one second
    ser.set_buffer_size(rx_size = inputBufferSize)

    global plotdata1
    length1 = int(audio_window * audio_samplerate / (1000 * audio_downsample))
    plotdata1 = np.zeros((length1, len(audio_channels)))

    # spikerbox is plotdata2
    global plotdata2
    length2 = int(max_time * spiker_samplerate / (spiker_downsample)) # note: maybe incorrect -> window may not be 2 seconds (Keegan)
    plotdata2 = np.zeros((length2, len(audio_channels)))

    # stream takes audio input data through audio_callback
    stream = sd.InputStream(
        device=audio_device, blocksize=audio_blocksize,
        channels=max(audio_channels), samplerate=audio_samplerate,
        callback=audio_callback)

    stream.start()

    time.sleep(0.1)

    recording_started = False

    calibration_spiker_data = []
    calibration_audio_data = []

    instructions_eye = """Welcome to Calibration.
    \nMomentarily, you will be asked to complete a RIGHT eye movement, 5 times.
    \nEach one separated and counted down so you know when to move.
    \nEach will be recorded for {t_stop} seconds.
    """.format(t_stop = t_stop)

    instructions_mic = """Now, you will be asked to produce the SOUND which you use to shoot.
    \nYou will be asked to make this sound 5 times.
    \nEach will be recorded for {t_stop} seconds.
    """.format(t_stop = t_stop)


    for i in range(2): # one for left looking one for microphone whistle peak finding.
        if i == 0:
            print(instructions_eye)
        elif i == 1:
            print(instructions_mic)
        for j in range(CALIBRATION_NUMBER):
            if (i == 0):
                while True:
                    # run an infinite loop of callback and processing functions

                    # update the mic plotdata
                    calibration_update_mic_plotdata()

                    # input an inputBufferSize of spikerbox data
                    spiker_callback(ser, inputBufferSize)

                    # Update the spiker plotdata
                    calibration_update_spiker_plotdata()

                    # print out confirmation that recording has started:
                    if not recording_started:
                        print("3...", end="", flush=True)
                        time.sleep(1)
                        print("2...", end="", flush=True)
                        time.sleep(1)
                        print("1...", end="", flush=True)
                        time.sleep(1)
                        print("Recording started.", flush=True)
                        t_i = time.time()
                        recording_started = True

                    # only do for `t_stop` seconds
                    if time.time() - t_i > t_stop:
                        print("Recording finished.\n")
                        calibration_spiker_data.append(spiker_full_data)
                        recording_started = False
                        break
            else:
                while True:
                    # run an infinite loop of callback and processing functions

                    # update the mic plotdata
                    calibration_update_mic_plotdata()

                    # input an inputBufferSize of spikerbox data
                    spiker_callback(ser, inputBufferSize)

                    # Update the spiker plotdata
                    calibration_update_spiker_plotdata()

                    # print out confirmation that recording has started:
                    if not recording_started:
                        print("3...", end="", flush=True)
                        time.sleep(1)
                        print("2...", end="", flush=True)
                        time.sleep(1)
                        print("1...", end="", flush=True)
                        time.sleep(1)
                        print("Recording started.", flush=True)
                        t_i = time.time()
                        recording_started = True

                    # only do for `t_stop` seconds
                    if time.time() - t_i > t_stop:
                        print("Recording finished.\n")
                        calibration_audio_data.append(audio_full_data)
                        recording_started = False
                        break

    # Call find_peak_freq on calibration mic data to find an average threshold to set as global.
    mic_threshold = np.mean([np.max(x) for x in calibration_audio_data])/2

    # Use wave calibration data to fix orientation of electrodes on head to match training data (A BIT BUGGY)
    simple_orientation = stats.mode([simple_classifier(x) for x in calibration_spiker_data])[0][0]

    return mic_threshold, simple_orientation


def extract_relevant_data(window, shift_length, factor):
    return window[len(window)-(shift_length*factor):]

def find_shortest_distance(a,b):
    smallest_dist = None
    for i in a[0]:
        for j in b[0]:
            dist = abs(i - j)
            if smallest_dist == None or dist < smallest_dist[0]:
                smallest_dist = (dist, i, j)
    return smallest_dist[1], smallest_dist[2]

def simple_classifier(data):
    pos_peaks = np.where(data == np.amax(data))
    neg_peaks = np.where(data == np.amin(data))
    pos_peak, neg_peak = find_shortest_distance(pos_peaks, neg_peaks)
    if pos_peak < neg_peak:
        return 0
    elif pos_peak >= neg_peak:
        return 1



########
# MAIN # ============================================================================================================
########

def main():

    # LOGISTIC REGRESSION CLASSIFIER LOADING/BUILDING
    model_filename = "model_waves.sav"
    scaler_filename = "scaler_waves.sav"
    
    if os.path.exists(model_filename) and os.path.exists(scaler_filename):      
        classifier = pickle.load(open(model_filename, 'rb'))
        scaler = pickle.load(open(scaler_filename, 'rb'))   
    else:
        classifier, scaler = build_classifier(f"new_training_data/spiker_waves", show_acc=True)

    # Variables
    run = True
    FPS = 120
    level = 0
    lives = 3
    main_font = pygame.font.SysFont("comicsans", 50)
    lost_font = pygame.font.SysFont("comicsans", 60)
    enemies = []
    wave_length = 5
    enemy_vel = 1
    player_vel = 130
    laser_vel = 30
    player = Player(300, 630)
    clock = pygame.time.Clock()
    lost = False
    lost_count = 0
    lockout = 0
    moved = False
    
    # Global variables
    global ser
    global simple_orientation
    global WIN
    global BG

    # list the audio devices
    if list_devices:
        # print the sound devices available
        print(sd.query_devices())
        exit()

    # if there is no samplerate given in commandline, assume the default samplerate for device
    audio_samplerate = 44100
    if audio_samplerate is None:
        device_info = sd.query_devices(audio_device, 'input')
        audio_samplerate = device_info['default_samplerate']

    # take continuous data stream
    ser.timeout = inputBufferSize/20000.0  # set read timeout, 20000 is one second
    ser.set_buffer_size(rx_size = inputBufferSize)

    global plotdata1
    length1 = int(audio_window * audio_samplerate / (1000 * audio_downsample))
    plotdata1 = np.zeros((length1, len(audio_channels)))

    # spikerbox is plotdata2
    global plotdata2
    length2 = int(max_time * spiker_samplerate / (spiker_downsample)) # note: maybe incorrect -> window may not be 2 seconds (Keegan)
    plotdata2 = np.zeros((length2, len(audio_channels)))

    # stream takes audio input data through audio_callback
    stream = sd.InputStream(
        device=audio_device, blocksize=audio_blocksize,
        channels=max(audio_channels), samplerate=audio_samplerate,
        callback=audio_callback)

    # Start collecting data from spikerbox
    stream.start()

    # Allow some starting wiggle room ;)
    time.sleep(0.1)

    # redraw the window each frame
    def redraw_window():
        WIN.blit(BG, (0,0))
        # draw text
        lives_label = main_font.render(f"Lives: {lives}", 1, (255,255,255))
        level_label = main_font.render(f"Level: {level}", 1, (255,255,255))

        WIN.blit(lives_label, (10, 10))
        WIN.blit(level_label, (WIDTH - level_label.get_width() - 10, 10))

        for enemy in enemies:
            enemy.draw(WIN)

        player.draw(WIN)

        if lost:
            lost_label = lost_font.render("You Lost!!", 1, (255,255,255))
            WIN.blit(lost_label, (WIDTH/2 - lost_label.get_width()/2, 350))

        pygame.display.update()



    # MAIN LOOP
    while run:
        
        # To delay consecutive movements 
        if lockout % 5 == 0:
            moved = False
        
        clock.tick(FPS)
        redraw_window()

        # Determine game status
        if lives <= 0 or player.health <= 0:
            lost = True
            lost_count += 1

        # Terminate game accordingly
        if lost:
            if lost_count > FPS * 3:
                run = False
            else:
                continue

        # Increase level when enemies are defeated
        if len(enemies) == 0:
            level += 1
            wave_length += 5
            for i in range(wave_length):
                enemy = Enemy(300+random.randint(-2, 2)*player_vel, random.randrange(-1500, -100), random.choice(["red", "green"]))
                enemies.append(enemy)

        # Collect more data
        spiker_callback(ser, inputBufferSize)

        # Update the spikerbox stored window of data
        new_data_len = update_spiker_plotdata()
        
        global shift_length
        if new_data_len != 0:
            shift_length = new_data_len

        # Only necessary for logistic regression
        wave = convert_format(plotdata2, spiker_samplerate) 

        global simple_orientation

        # UNCOMMENT FOR SIMPLE CLASSIFIER
        L_R = simpleclass(data=plotdata2, shift_length=shift_length, simple_orientation=simple_orientation)

        # UNCOMMENT FOR REGRESSION CLASSIFIER
        # L_R = int(predict(wave=wave, classifier=classifier, standard_scaler=scaler)[0])
        # if L_R == 0:
        #     L_R = 'L'
        # elif L_R == 1:
        #     L_R = 'R'
        # else:
        #     L_R = None


        ## Microphone processing
        # window update
        update_mic_plotdata()
        # read window and classify
        shoot = classify_mic() # bool

        if shoot:
            player.shoot()

        # Move player according to classification and lock status
        if (L_R == 'L' and player.x - player_vel > 0) and not moved:
            player.x -= (player_vel) 
            moved = True
        elif (L_R == 'R' and player.x + player_vel + player.get_width() < WIDTH) and not moved:
            player.x += (player_vel)
            moved = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a and player.x - player_vel > 0:
                    player.x -= player_vel
                if event.key == pygame.K_d and player.x + player_vel + player.get_width() < WIDTH:
                    player.x += player_vel
                if event.key == pygame.K_SPACE:
                    player.shoot()
                if event.key == pygame.K_q:
                    pygame.quit()

        for enemy in enemies[:]:
            enemy.move(enemy_vel)
            enemy.move_lasers(laser_vel, player)

            if random.randrange(0, 2*60) == 1:
                enemy.shoot()

            if collide(enemy, player):
                player.health -= 10
                enemies.remove(enemy)
            elif enemy.y + enemy.get_height() > HEIGHT:
                lives -= 1
                enemies.remove(enemy)

        player.move_lasers(-laser_vel, enemies)
        
        lockout += 1

def game_menu():
    
    # GAME WINDOW INITIALISATION
    pygame.font.init()
    global WIN
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Space Shooter Tutorial")
    
    # Background
    global BG
    BG = pygame.transform.scale(pygame.image.load(os.path.join("assets", "background-black.png")), (WIDTH, HEIGHT))

    # title font initalised
    title_font = pygame.font.SysFont("comicsans", 70)

    run = True
    while run:
        WIN.blit(BG, (0,0))
        title_label = title_font.render("Press the mouse to begin...", 1, (255,255,255))
        WIN.blit(title_label, (WIDTH/2 - title_label.get_width()/2, 350))
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                main()
    pygame.quit()


if __name__ == "__main__":
    welcome_menu()
