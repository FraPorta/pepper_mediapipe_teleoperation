import argparse
import numpy as np
import os
import matplotlib.pyplot as plt

from os.path import isfile, join

## class PlotAngles
#
# loads a list of csv files from the specified folder (path) and plot them using matplotlib
class PlotAngles:
    def __init__(self, path, n, init, trim_s, trim_e):
        self.path = path
        self.n = n
        self.init = init
        self.end = self.init +self.n
        
        self.trim_end = trim_e
        self.trim_start = trim_s
        
        self.map = {1: (1,1),
                    2: (2,1),
                    3: (2,2),
                    4: (2,2),
                    5: (3,2),
                    6: (3,2),
                    7: (4,2),
                    8: (4,2),
                    9: (5,2),
                    10:(5,2)} 
                    
        self.pos = {0: (0,0),
                    1: (0,1),
                    2: (1,0),
                    3: (1,1),
                    4: (2,0),
                    5: (2,1),
                    6: (3,0),
                    7: (3,1),
                    8: (4,0),
                    9: (4,1)
                    }
        
        self.green = "#14c417"
        self.orange = "#e37632"
        self.blu = "#215fa6" 
        
    ##  function plot_data
    #
    #   Plot raw and filtered angles at the end of the session
    def plot_data(self, axs, raw_data, filt_data, robot_data, name, time_samples, pos):
        # Plot time signals (Raw and filtered)
        data = np.array(raw_data[self.trim_start:self.trim_end:])
        
        if len(raw_data) > len(filt_data):
            filt_data.append(0.0)
        data_filt = np.array(filt_data[self.trim_start:self.trim_end:])
        
        if len(raw_data) > len(robot_data):
            robot_data.append(0.0)
        data_robot = np.array(robot_data[self.trim_start:self.trim_end:])
        
        time_samples = time_samples[self.trim_start:self.trim_end:]
        
        if len(time_samples) == len(data):
            axs[pos[0], pos[1]].plot(time_samples, data, color=self.green)
            axs[pos[0], pos[1]].set(xlabel='time [s]', ylabel='Angle [rad]')
            axs[pos[0], pos[1]].set_title(name[5:8:])
            
        if len(time_samples) == len(data_filt):
            axs[pos[0], pos[1]].plot(time_samples, data_filt, color=self.blu)
            # axs[pos_x, pos_y].legend(['signal', 'filtered'])
        
        if len(time_samples) == len(data_robot):
            axs[pos[0], pos[1]].plot(time_samples, data_robot, color=self.orange, linestyle='dashed')
            axs[pos[0], pos[1]].legend(['signal', 'filtered', 'robot'])
            # axs[pos[0], pos[1]].legend(['Human', 'Robot'])
    
    ##  function plot_data
    #
    #   Plot raw and filtered angles at the end of the session
    def plot_data_2(self, axs, raw_data, filt_data, robot_data, name, time_samples, pos):
        # Plot time signals (Raw and filtered)
        data = np.array(raw_data[self.trim_start:self.trim_end:])
        
        if len(raw_data) > len(filt_data):
            filt_data.append(0.0)
        data_filt = np.array(filt_data[self.trim_start:self.trim_end:])
        
        if len(raw_data) > len(robot_data):
            robot_data.append(0.0)
        data_robot = np.array(robot_data[self.trim_start:self.trim_end:])
        
        time_samples = time_samples[self.trim_start:self.trim_end:]
        
        if len(time_samples) == len(data):
            axs[pos[1]].plot(time_samples, data, color=self.green)
            axs[pos[1]].set(xlabel='time [s]', ylabel='Angle [rad]')
            axs[pos[1]].set_title(name[5:8:])
            
        if len(time_samples) == len(data_filt):
            axs[pos[1]].plot(time_samples, data_filt, color=self.blu)
            # axs[pos_x, pos_y].legend(['signal', 'filtered'])
        
        if len(time_samples) == len(data_robot):
            axs[pos[1]].plot(time_samples, data_robot, color=self.orange, linestyle='dashed')
            axs[pos[1]].legend(['signal', 'filtered', 'robot'])
            # axs[pos[1]].legend(['Human', 'Robot'])
    
    ##  function plot_data
    #
    #   Plot raw and filtered angles at the end of the session
    def plot_data_1(self, axs, raw_data, filt_data, robot_data, name, time_samples):
        # Plot time signals (Raw and filtered)
        data = np.array(raw_data[self.trim_start:self.trim_end:])
        
        if len(raw_data) > len(filt_data):
            filt_data.append(0.0)
        data_filt = np.array(filt_data[self.trim_start:self.trim_end:])
        
        if len(raw_data) > len(robot_data):
            robot_data.append(0.0)
        data_robot = np.array(robot_data[self.trim_start:self.trim_end:])
        
        time_samples = time_samples[self.trim_start:self.trim_end:]
        
        if len(time_samples) == len(data):
            axs.plot(time_samples, data, color=self.green)
            axs.set(xlabel='time [s]', ylabel='Angle [rad]')
            axs.set_title(name[5:8:])
            
        if len(time_samples) == len(data_filt):
            axs.plot(time_samples, data_filt, color=self.blu)
            # axs[pos_x, pos_y].legend(['signal', 'filtered'])
        
        if len(time_samples) == len(data_robot):
            axs.plot(time_samples, data_robot, color=self.orange, linestyle='dashed')
            axs.legend(['signal', 'filtered', 'robot'])
            # axs.legend(['Human', 'Robot'])            
            
    ## method run
    #
    # load and plot data from the specified folder
    def run(self):
        # Loads list of files in the folder
        files_list = [f for f in os.listdir(self.path) if isfile(join(self.path, f)) and 'data' in f]
        files_list.reverse()
        
        if self.n > 10:
            self.n = 10
        if self.init > 9:
            self.init = 9
            
        files_list = files_list[self.init:self.end:]
        
        # Create figure with 10 subplots
        fig, axs = plt.subplots(self.map.get(self.n)[0],self.map.get(self.n)[1])
        fig.suptitle('Joints angles')

        for i, f in enumerate(files_list):
            data = np.loadtxt(self.path + '/' + f,
                            delimiter =", ")
            
            name = files_list[i]
            data_raw = data[0, :]
            data_filt = data[1, :]
            data_robot = data[2, :]
            time_samples = data[3, :]
            
            if self.trim_end > len(data_raw):
                self.trim_end = len(data_raw)
            
            # Plot i-th joint angles
            if self.n > 2:
                self.plot_data(axs, data_raw, data_filt, data_robot, name, time_samples, self.pos.get(i))
            elif self.n == 2:
                self.plot_data_2(axs, data_raw, data_filt, data_robot, name, time_samples, self.pos.get(i))
            else:
                self.plot_data_1(axs, data_raw, data_filt, data_robot, name, time_samples)
                
        
        print("Showing angles plots, close to terminate the program.")
        plt.subplots_adjust(wspace=0.28,hspace=0.42, top=0.89, left=0.11, bottom=0.09, right=0.96)
        plt.show()
        
def all_subdirs_of(b='angles_data/'):
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd): result.append(bd)
    return result
            
        
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="21_09_2021_12-02-03_Milla3",
                        help="Insert name of the folder where the angles are stored in a csv file in the 'angles_data' folder")
    parser.add_argument("--n_plots", type=int, default="2",
                        help="Insert number of angles you want to display")
    parser.add_argument("--init", type=int, default="8",
                        help="Insert number of the first angle you want to display")


    args = parser.parse_args()
    # path = "angles_data/" + args.path
    path = max(all_subdirs_of(), key=os.path.getmtime)
    # print(latest_subdir)
    # path = "angles_data/" + "11_05_2022_17-14-23"
    
    n_subplots = args.n_plots
    init = args.init
    trim_start = 0
    trim_end = 10000
    pa = PlotAngles(path, n_subplots, init, trim_start, trim_end)
    pa.run()