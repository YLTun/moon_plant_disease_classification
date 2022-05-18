import pickle
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
# from sklearn import metrics
from pynvml import *

# Util for saving objects in pickle format.
def save_pickle(file_path, obj):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


# Util for loading objects from pickle format.
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def print_separator(text='', seperator_length=20):
    print('\n', '='*seperator_length + ' ' + text + ' ' + '='*seperator_length + '\n')


# Util for printing result at the end of each round.
def print_result(performance_log):
    print('train     -              loss :    {:.4f}          acc:    {:.4f}'.format(performance_log['train_loss'][-1], performance_log['train_acc'][-1]))
    print('valid     -              loss :    {:.4f}          acc:    {:.4f}'.format(performance_log['valid_loss'][-1], performance_log['valid_acc'][-1]))
    print()


# For plotting model history.
def save_history_plot(history_log, plot_config):
    
    plt.figure()
    for attribute, label in zip(plot_config['attributes'], plot_config['labels']):
        plt.plot(history_log[attribute], label=label)
    plt.title(plot_config['title'])
    plt.xlabel(plot_config['xlabel'])
    plt.ylabel(plot_config['ylabel'])
    plt.grid(True, linestyle='-.')
    plt.legend()

    if plot_config['save_dir'] != None:
        plt.savefig(plot_config['save_dir'], dpi=200, bbox_inches='tight')
    if plot_config['show_img']:
        plt.show()

    plt.close('all')


# Compute accuracy.
def compute_accuracy(y_batch, y_pred):
    _, predicted = torch.max(y_pred, 1)
    accuracy = (predicted == y_batch).sum().item() / len(y_batch)
    # accuracy = metrics.accuracy_score(y_batch.cpu(), predicted.cpu())
    return accuracy


# Helper for computing metrics at each epoch.
class MeanMetric():
    
    def __init__(self):
        self.total = np.float32(0)
        self.count = np.float32(0)

    def update_state(self, value):
        self.total += value
        self.count += 1
        
    def result(self):
        if self.count > 0:
            return self.total / self.count
        else:
            return np.nan

    def reset_state(self):
        self.total = np.float32(0)
        self.count = np.float32(0)


# Plotting settings
LOSS_PLOT_CONFIG = {
  'attributes': ('train_loss', 'valid_loss'),
  'labels': ('train', 'valid'),
  'title': 'Loss',
  'xlabel': 'rounds',
  'ylabel': 'loss',
  'save_dir': None,   
  'show_img': False,
}

ACC_PLOT_CONFIG = {
    'attributes': ('train_acc', 'valid_acc'),
    'labels': ('train', 'valid'),
    'title': 'Accuracy',
    'xlabel': 'rounds',
    'ylabel': 'accuracy',
    'save_dir': None,
    'show_img': False,
}

# For plotting model history.
def save_history_plot(history_log, plot_config):
    
    plt.figure()
    for attribute, label in zip(plot_config['attributes'], plot_config['labels']):
        plt.plot(history_log[attribute], label=label)
    plt.title(plot_config['title'])
    plt.xlabel(plot_config['xlabel'])
    plt.ylabel(plot_config['ylabel'])
    plt.grid(True, linestyle='-.')
    plt.legend()

    if plot_config['save_dir'] != None:
        plt.savefig(plot_config['save_dir'], dpi=200, bbox_inches='tight')
    if plot_config['show_img']:
        plt.show()

    plt.close('all')


# Helpers for logging model performance.
def get_performance_loggers(metric_keys = {'train_loss', 'train_acc', 'valid_loss', 'valid_acc'}):
	performance_dict, performance_log = dict(), dict()
	for key in metric_keys:
	    performance_dict[key] = MeanMetric()
	    performance_log[key] = list()
	return performance_dict, performance_log


# GPU memory allocation workaround.
def allocate_gpu_memory(mem_amount=10504699904):
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'total    : {info.total}')
    print(f'free     : {info.free}')
    print(f'used     : {info.used}')

    block_mem = int(info.free * 0.8) // (32 // 8)
    block_mem = int(block_mem * 0.8)
    block_mem

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.rand(block_mem, dtype=torch.float32).to(device)
    x = torch.rand(1)
    del x

    nvmlShutdown()


def save_notes(file_path, notes):
    with open(file_path, 'w') as f:
        f.write(notes)


# Util for showing results.
def get_mean_std(num_list):
    print('mean:{:.2f}'.format(np.mean(num_list)))
    print('std:{:.2f}'.format(np.std(num_list)))
