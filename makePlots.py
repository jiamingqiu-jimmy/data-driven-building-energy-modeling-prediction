import numpy as np
import pickle
import matplotlib.pyplot as plt


def plot_Data_hyperparameters(records):
    plt.figure(figsize=(13, 10))
    plt.ion()

    plt.subplot(2,2,1)
    lrs = [.1, .01, .001, .0001]
    offsets = [(-13,14),(-20,10),(-50,10),(-9,-12)]
    for idx_h1 in range(2):
        for idx_h2 in range(2):
            plt.subplot(2,2,(2*idx_h1 +idx_h2)+1)
            for idx_lr in range(4):
                record_idx = (2*idx_h1 +idx_h2) + 4 * idx_lr
                # print(record_idx)
                record = records[f'arr_{record_idx}']
                plt.plot(range(len(record)), record, label='{}'.format(lrs[idx_lr]))
                plt.yscale('log')
                # plt.xlabel('epochs')
                plt.annotate('%0.2f' % record.min(), xy=(len(record),record.min()), xytext= offsets[idx_lr], xycoords=('data', 'data'), textcoords='offset pixels')
                plt.ylabel('MSELoss')
                ax = plt.gca()
                ax.set_ylim([5*10e1,5*10e4])
                plt.title('Hidden Layer Amounts: {} {}'.format((idx_h1+1)*10,(idx_h2+1)*10))
                plt.legend(title='Learning rates')
    plt.show()
    plt.savefig('plots\\loss_graph_hyperparameters.png')

def plot_Data_intervals(records):
    plt.figure(figsize=(11, 8))
    plt.ion()

    intervals = [30,60,120]
    correction = [4,2,1]
    offsets = [(-15,10),(-1,10),(-18,10)]
    for idx_interval in range(3):
        # print(record_idx)
        record = records[f'arr_{idx_interval}']#/correction[idx_interval]
        if idx_interval == 1:
            record = record[0:-2]
        plt.plot(range(len(record)), record, label='{}'.format(intervals[idx_interval]))
        plt.yscale('log')
        plt.xlabel('epochs')
        plt.annotate('%0.2f' % record.min(), xy=(len(record),record.min()), xytext= offsets[idx_interval], xycoords=('data', 'data'), textcoords='offset pixels')
        plt.ylabel('MSELoss')
        ax = plt.gca()
        # ax.set_ylim([5*10e1,5*10e4])
        plt.title('Plotting with different data intervals')
        plt.legend(title='Data Intervals')
    plt.show()
    plt.savefig('plots\\loss_graph_intervals.png')



if __name__ == '__main__':
    records_hyperparemeters = np.load('runtime data\\all_hyperparameters_loss_records.npz')
    plot_Data_hyperparameters(records_hyperparemeters)
    records_intervals = np.load('runtime data\\all_intervals_loss_records.npz')
    plot_Data_intervals(records_intervals)
    pass