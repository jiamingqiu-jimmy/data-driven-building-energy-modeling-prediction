#!/usr/bin/env python
# coding: utf-8

# Graph neural network
# Inputs:
# Nodes of all the rooms
# Outputs:
# 
# Important rules:
# temp = Temperature
# tmp = Temporary
# w = weight
# 

# In[5]:


import torch
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import pickle


# from torch_geometric.nn import GCNConv
# from torch_geometric.datasets import Planetoid


# In[6]:


class Model(nn.Module):
    """ 
    Graph Network model connecting all the rooms in a building
        Learns how a heat radiates between rooms, given each room's temperature.
    """

    def __init__(self, num_rooms, INNdim_list, room_order, loss_func = nn.MSELoss()) -> None:
        """Init

        Args:
            num_rooms (int): Amount of rooms
            INNdim_list (list of int): dimensions for each layer of individual network

            n amount of rooms x n amount of rooms 

        """
        super(Model, self).__init__()

        self.GNN = GNNModel(num_rooms, room_order, loss_func)

        self.INN = INNModel(INNdim_list, loss_func)

        self.weight_INN = nn.Parameter(torch.ones(1)*.5)
        self.weight_GNN = nn.Parameter(torch.ones(1)*.5)

    def train(self, dataGNN, dataINN, y, epochs, epsilon, max_runtime, outfile, visible = False):
        """
        Loss Function: Actual temp - Predicted Temp
        Predicted Temp = af + bg

        Returns:
            _type_: _description_
        """

        # data prep
        loss_record = []

        t = 0

        total_start = time.time()

        loss_improv = 1
        stop_next = False

        dims = dataINN.shape

        # plt.figure(figsize=(12, 6))
        # plt.ion()
        # plt.show()
        while True:
            # plt.clf()

            epoch_start = time.time()

            loss_sum = 0
            for datum_idx in range(dims[0]-1):
                optimizer.zero_grad()
                cur_temp = torch.clone(dataGNN[datum_idx,:])
                features = torch.clone(dataINN[datum_idx,:,:])
                new_temp = self.predict(cur_temp, features)
                true_temp = torch.clone(dataINN[datum_idx+1,:,1])
                loss = loss_func(new_temp, true_temp)
                loss_sum += loss.cpu().detach()
                if datum_idx == 0 and visible:
                    print('Epoch Datum Idx:')
                if datum_idx % 100 == 0 and visible:
                    print(datum_idx, end=' ')
                loss.backward()
                optimizer.step()
                if datum_idx > 5:
                    break

            if t >= 1:
                loss_improv = loss_sum/loss_record[-1]
            loss_record.append(loss_sum)
            t += 1

            if visible:
                print('')
                self.print_loss(loss_sum, t, epoch_start, loss_improv)


            # plt.plot(range(len(loss_record)),  loss_record, label= 'Total Loss: {}'.format(t))
            # plt.yscale('log')
            # plt.legend()
            #
            # plt.draw()
            # plt.pause(.001)
            # plt.savefig('images\\loss_graph2.png')

            if 1- loss_improv < epsilon:
                if stop_next:
                    break
                stop_next = True
            elif time.time() - total_start >= max_runtime or (t >= epochs and epochs !=0):
                break
            else:
                stop_next = False


        torch.save(model.state_dict(), outfile)
        # np.savez(f'runtime data\\loss records.npz', loss_record)
        # print('Model saved')
        print('Final Loss: {}'.format(loss_record[-1]))
        print('Total time taken: {}'.format(time.time() - total_start))
        return loss_record

    def predict(self, temp_start, features):
        dims = features.shape
        gnn_temp = self.GNN.predict(temp_start)
        inn_temp = torch.zeros(dims[0])
        for room in range(dims[0]):
            inn_temp[room] = self.INN.predict(features[room])
        inn_temp = inn_temp.to(gnn_temp.device)
        # new_temp = temp_start + self.weight_GNN * (gnn_temp - temp_start) + self.weight_INN * (inn_temp - temp_start )
        new_temp = self.weight_GNN * gnn_temp + self.weight_INN * inn_temp
        return new_temp

    def print_loss(self, loss, t, start, loss_improv):
        print('-' * 65)
        print('Epochs: {}, Time: {}, MSE Loss: {}'.format(t, time.time()-start, loss))
        print('Loss Improvement: {}'.format(loss_improv))
        print('-' * 65)

class GNNModel(torch.nn.Module):
    """GNN_Node
        This represents a room
    """

    def __init__(self, num_rooms, room_order, loss_func):
        """Node initialization

        Args:
            temperature (_type_): _description_
        """
        super(GNNModel, self).__init__()

        self.adjacency_matrix = nn.Parameter(self.create_matrix(num_rooms, room_order))
        # need vector for individual rooms

        self.self_temp_w = nn.Parameter(torch.ones(num_rooms)*.8)
        self.other_temp_w = nn.Parameter(torch.ones(num_rooms)*.2)
        self.loss_func = loss_func
        self.relu = nn.ReLU()

    def create_matrix(self, num_rooms, room_order):
        matrix = np.ones((num_rooms, num_rooms)) * -1
        room_dict = dict()
        for idx, room in enumerate(room_order):
            room_dict[room] = idx

        matrix[room_dict['403'],room_dict['405']] = 1
        matrix[room_dict['461'],room_dict['462']] = 1
        matrix[room_dict['461'],room_dict['463']] = 1
        matrix[room_dict['462'],room_dict['463']] = 1
        matrix[room_dict['448'],room_dict['450']] = 1
        matrix[room_dict['428'],room_dict['429']] = 1
        matrix[room_dict['302'],room_dict['304']] = 1
        matrix[room_dict['386'],room_dict['387']] = 1
        matrix[room_dict['348'],room_dict['350']] = 1
        matrix[room_dict['340'],room_dict['345']] = 1
        matrix[room_dict['361'],room_dict['362']] = 1
        matrix[room_dict['361'],room_dict['363']] = 1
        matrix[room_dict['362'],room_dict['363']] = 1
        matrix[room_dict['328'],room_dict['329']] = 1
        matrix[room_dict['329'],room_dict['330']] = 1
        matrix[room_dict['330'],room_dict['336']] = 1
        matrix[room_dict['280'],room_dict['281']] = 1
        matrix[room_dict['221'],room_dict['223']] = 1
        matrix[room_dict['261'],room_dict['264']] = 1
        matrix[room_dict['264'],room_dict['268']] = 1
        matrix[room_dict['245'],room_dict['247']] = 1
        matrix[room_dict['104'],room_dict['105']] = 1
        matrix[room_dict['114'],room_dict['120']] = 1
        matrix[room_dict['120'],room_dict['121']] = 1
        matrix[room_dict['121'],room_dict['122']] = 1

        matrix += np.transpose(matrix) + 1
        # matrix /= 25
        matrix = torch.from_numpy(matrix.astype(np.float32))
        # matrix = (torch.ones(num_rooms, num_rooms) - torch.eye(num_rooms))/num_rooms
        return matrix

    def ensure_self_zero(self):
        optimizer.zero_grad()
        size = self.adjacency_matrix.shape[0]
        loss_func_zero = nn.MSELoss()
        loss = 100 * loss_func_zero(torch.zeros(size).to(self.adjacency_matrix.device), torch.diagonal(self.adjacency_matrix))
        loss.backward()
        optimizer.step()

    def predict(self, data):
        aggregate_temp = torch.matmul(self.relu(self.adjacency_matrix), data)
        final_temp = self.self_temp_w * data + self.other_temp_w * aggregate_temp
        return final_temp


class INNModel(torch.nn.Module):
    """
    Room Network model:
        learning how the features of a room (AC, other stuff) affects the current temperature in a room
    Input training data:
    Output: Model
    """

    def __init__(self, dim_list, loss_func) -> None:
        """
        Initialize the Room Network Model
        We'll change these inits later, specifics on the layers later.

        Args:
            features (list): Each entry in the list is a feature value of the room
        """
        super(INNModel, self).__init__()
        self.linear_list = nn.ModuleList([nn.Linear(dim_list[i], dim_list[i + 1]) for i in range(len(dim_list) - 1)])
        self.loss_func = loss_func


    def predict(self, features):
        """
        Given the features, return some expected temperature
        """
        # room_temp = self.model.fit(features)
        # return room_temp
        values = features
        for layer_num in range(len(self.linear_list)):
            values = self.linear_list[layer_num](values)
        assert len(values) == 1, 'not proper INN output'
        return values


# Data
def get_data(interval):
    dataINN = torch.from_numpy(np.load(f'preprocessing_output\\{str(interval)}T\\features_rooms_training_{str(interval)}T.npy').astype(np.float32))
    yINN = torch.from_numpy(np.load(f'preprocessing_output\\{str(interval)}T\\features_rooms_training_{str(interval)}T.npy').astype(np.float32)[:,:,1])
    dataGNN = torch.from_numpy(np.load(f'preprocessing_output\\{str(interval)}T\\temps_training_{str(interval)}T.npy').astype(np.float32))
    return dataINN, yINN, dataGNN

rooms = np.load(f'preprocessing_output\\merged_rooms_list.npy')

time_interval = 120
dataINN, yINN, dataGNN =get_data(time_interval)

# Device selection
if torch.cuda.is_available() and 1 == 0:
    device = torch.device("cuda")
    dataINN = dataINN.cuda()
    yINN = yINN.cuda()
    dataGNN = dataGNN.cuda()
else:
    device = torch.device("cpu")

print(device, end='\n\n')

# Hyperparameters
epochs = 0
stop_epsilon = 0.0001
max_runtime = 7200
num_rooms = dataINN.shape[1]
network_layer_dims = [dataINN.shape[2], 20, 20, 1]
loss_func = nn.MSELoss()
learning_rate = .001

outfile = f'runtime data\\model_save.pt'

if __name__ == '__main__':
    print(device)
    test_all_hyperparameters = False
    test_all_intervals = True

    if test_all_hyperparameters:
        all_loss = np.zeros((4,2,2))
        all_loss_records = []
        for idx_lr, learning_rate in enumerate([.1, .01, .001, .0001]):
            for idx_h1, hidden_layer_1 in enumerate([10,20]):
                for idx_h2, hidden_layer_2 in enumerate([10,20]):
                    network_layer_dims[1] = hidden_layer_1
                    network_layer_dims[2] = hidden_layer_2
                    print('Learning rate: {}, Hidden layers: {} {}'.format(learning_rate, hidden_layer_1, hidden_layer_2))
                    model = Model(num_rooms, network_layer_dims, rooms, loss_func)
                    model = model.to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                    loss_record = model.train(dataGNN, dataINN, yINN, epochs,stop_epsilon, max_runtime, outfile, visible=True)
                    all_loss[idx_lr, idx_h1, idx_h2] = loss_record[-1]
                    all_loss_records.append(loss_record)

                print('\n')

        np.save('runtime data\\all_hyperparameters_loss_final', all_loss)
        np.savez('runtime data\\all_hyperparameters_loss_records', *all_loss_records)

    elif test_all_intervals:
        all_loss_records = []
        time_intervals = [30,60,120]
        for interval in time_intervals:
            dataINN, yINN, dataGNN = get_data(interval)
            print('Data interval: {}'.format(interval))
            model = Model(num_rooms, network_layer_dims, rooms, loss_func)
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            loss_record = model.train(dataGNN, dataINN, yINN, epochs, stop_epsilon, max_runtime, outfile, visible=True)
            all_loss_records.append(loss_record)
            print('\n')
        np.savez('runtime data\\all_intervals_loss_records', *all_loss_records)

    else:
        model = Model(num_rooms, network_layer_dims, rooms, loss_func)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss = model.train(dataGNN, dataINN, yINN, epochs, stop_epsilon, max_runtime, outfile, visible=True)
        np.save('runtime data\\loss.npz', loss)

