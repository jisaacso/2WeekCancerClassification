import pandas as pd
import numpy as np
import sklearn
import os

'''
def readCsvFiles(file_path="."):
    """ Reads in CSV files and creates a list of DataFrames"""
    data_frames_list = []
    for dir_path, dir_names, file_names in os.walk(file_path):
        data_class = dir_path[dir_path.rfind('/') + 1:]
        for file_name in file_names:
            # Read the file data and create lists of Pandas DataFrames
            file_handle = dir_path + '/' + file_name
            data_frame = pd.read_csv(file_handle, names=[
                                     "time",
                                     "acceleration_x",
                                     "acceleration_y",
                                     "acceleration_z",
                                     "acceleration_magnitude",
                                     "magneticfield_x",
                                     "magneticfield_y",
                                     "magneticfield_z",
                                     "magneticfield_magnitude",
                                     "useracceleration_x",
                                     "useracceleration_y",
                                     "useracceleration_z",
                                     "gravity_x",
                                     "gravity_y",
                                     "gravity_z",
                                     "roll"
                                     "pitch",
                                     "yaw",
                                     "rotationrate_x"
                                     "rotationrate_y",
                                     "rotationrate_z", ])
            data_frame.data_class = data_class
            data_frames_list.append(data_frame)

    return data_frames_list

current_path = os.path.dirname(os.path.realpath(__file__))

file_path = current_path + "/sample_data/"

# Read DataFrames into list
data_frames_list = readCsvFiles(file_path=file_path)

# Now you can access the data you need for parameter extraction
# by using the DataFrames's meta-data, "data_class"

# For example
for data_frame in data_frames_list:
    if data_frame.data_class=="walk":
        print("This is a walking data.")
    else:
        print("This is NOT walking data.")

# More helpful functions
'''
def slidingWindow(sequence, windowSize=2, stepSize=1):
    """Returns a generator that will iterate through
    the defined chunks of input sequence.  Input sequence
    must be iterable.

    Parameters
    a  - a 1D array
    windowSize - the window size, in samples
    stepSize - the step size, in samples. If not provided, window and step size
             are equal.
    """

    # Verify the inputs
    try:
        it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((isinstance(windowSize, type(0))) and (isinstance(stepSize, type(0)))):
        raise Exception(
            "**ERROR** type(windowSize) and type(stepSize) must be int.")
    if stepSize > windowSize:
        raise Exception(
            "**ERROR** stepSize must not be larger than windowSize.")
    if windowSize > len(sequence):
        print("**ERROR** windowSize must not be larger than sequence length.")
        return None

    # Pre-compute number of chunks to emit
    numOfChunks = int((len(sequence) - windowSize) / stepSize) + 1

    outputWindows = []
    for i in range(0, numOfChunks * stepSize, stepSize):
        outputWindows.append(sequence[i:i + windowSize])

    return outputWindows

def smooth(x, window_len=3, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
            x: the input signal
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                    flat window will produce a moving average smoothing.

    output:
            the smoothed signal

    example:
    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError(
            "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y

def medianFilter(x, k=9):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros ((len (x), k), dtype=x.dtype)
    y[:,k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    return np.median (y, axis=1)




