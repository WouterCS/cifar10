import numpy as np
import matplotlib.pyplot as plt
from Tkinter import *

def plot_coarse(timeSeries, saveDirectory):
    plt.clf()
    plt.plot(timeSeries)
    plt.figure(num = 1, figsize = (20,20), dpi = 800)
    plt.ylabel('accuracy (%)')
    plt.grid(b=True, which='major', color='black', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='black', linestyle='--')
    plt.ylim(0,100)
    plt.savefig(saveDirectory + '/coarse_plot.png')
    
def plot_detailed(timeSeries, saveDirectory):
    timeSeries = timeSeries[-20:]
    plt.clf()
    plt.plot(timeSeries)
    plt.figure(num =1, figsize = (20,20), dpi = 800)
    plt.ylabel('accuracy (%)')
    plt.grid(b=True, which='major', color='black', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='black', linestyle='--')
    minimum = np.min(timeSeries)
    maximum = np.max(timeSeries)
    margin = 0.1 * (maximum - minimum)
    plt.ylim(np.max(minimum - margin, 0), np.min(maximum + margin, 100))
    plt.savefig(saveDirectory + '/detailed_plot.png')