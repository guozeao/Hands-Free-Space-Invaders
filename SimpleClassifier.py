""" Simple Classifier variables and functions.
Currently importing into spaceinvaders_streamer file, once fixed need to
move into the final code.
"""

import numpy as np

# simple classifier global variables
peaked = False
movement_buffer_lock = 0
iteration = 0
sd_sum = 0


def convert_to_L_R(choice, simple_orientation):
        """
        Takes a choice (0, 1, 2) and converts that into an L-R movement, or nothing.
        @choice             : int (0, 1, 2) output from predict()
        @simple_orientation : int (0, 1) which is what a LEFT movement is
                                recorded as from the calibration.
        @return L_R         : str ('L, 'R', '') Left-movement, Right-movement or
                                nothing.
        """
        L_R = ''
        choice = int(choice)
        simple_orientation = int(simple_orientation)

        if choice == 2:
            L_R = ''
            return L_R

        if simple_orientation == 1:
            if choice == 0:
                L_R = 'L'
            elif choice == 1:
                L_R = 'R'
        elif simple_orientation == 0:
            if choice == 1:
                L_R = 'L'
            elif choice == 0:
                L_R = 'R'

        return L_R

def simple_classifier(data, simple_orientation):
    choice = ''

    pos_peak = np.where(data == np.amax(data))[0]
    neg_peak = np.where(data == np.amin(data))[0]
    #print(f"POS PEAK: {pos_peak} and NEG PEAK: {neg_peak}")
    if max(pos_peak) < min(neg_peak):
        #print("LEFT MOVEMENT")
        choice = '0'
        L_R =convert_to_L_R(choice, simple_orientation)
    else:
        #print("RIGHT MOVEMENT")
        choice = '1'
        L_R = convert_to_L_R(choice, simple_orientation)
    return L_R

def extract_relevant_data(window, shift_length, factor):
    return window[len(window)-(shift_length*factor):]

def simpleclass(data, shift_length, simple_orientation):
    global peaked
    global movement_buffer_lock
    global interation
    global sd_sum
    new_data = data[len(data)-shift_length:]
    sd = np.std(new_data)
    avg = np.mean(new_data)
    if (sd > 20) and (sd < 1000):
        peaked = True
    if movement_buffer_lock == 4:
        if peaked:
            window = extract_relevant_data(data, shift_length, 4) # TO STORE 1 SECONDS WORTH OF DATA
            L_R = simple_classifier(window, simple_orientation)
            peaked = False
            return L_R
        movement_buffer_lock = 0

    movement_buffer_lock += 1
