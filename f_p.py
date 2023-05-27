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
import matplotlib as plt


# from torch_geometric.nn import GCNConv
# from torch_geometric.datasets import Planetoid


# In[6]:


class Model(nn.Module):
    """ 
    Graph Network model connecting all the rooms in a building
        Learns how a heat radiates between rooms, given each room's temperature.
    """

    def __init__(self, num_rooms, INNdim_list) -> None:
        """Init

        Args:
            num_rooms (int): Amount of rooms
            INNdim_list (list of int): dimensions for each layer of individual network

            n amount of rooms x n amount of rooms 

        """
        super(Model, self).__init__()

        self.GNN = GNNModel(num_rooms)

        self.INN = INNModel(INNdim_list)

        self.weight_INN = nn.Paramter(1)
        self.weight_GNN = nn.Paramter(1)

    def train(self, dataGNN, dataINN, y, epochs, stop_cond, visible = False):
        """
        Loss Function: Actual temp - Predicted Temp
        Predicted Temp = af + bg

        Returns:
            _type_: _description_
        """

        # data prep
        loss_record = []
        t = 0
        while True:

            lossGNN = self.GNN.train_gnn(dataGNN, dataINN, y, self.INN, self.weight_GNN, self.weight_INN)
            lossINN = self.INN.train_inn(dataGNN, dataINN, y, self.GNN, self.weight_GNN, self.weight_INN)
            loss_total = lossGNN + lossINN
            loss_record.append(loss_total)
            t += 1
            if visible:
                self.print_loss(loss_total, t)
            if loss_total/loss_record[-2] <= stop_cond or t > epochs:
                break
        plt.plot(loss_record)
        plt.show()

    def predict(self, temp_start, features):
        new_temp = temp_start
        temp_history = [new_temp]
        dims = features.shape
        full_features = torch.cat(torch.zeros(dims[0]),features)

        for t in range(dims[0]):
            # add temp to first feature slot
            full_features[t][0] = new_temp
            gnn_temp = self.GNN.predict(new_temp)
            inn_temp = torch.zeros(dims[1])
            for room in range(dims[1]):
                inn_temp[room] = self.INN.predict(features[room]).detach()
            new_temp += self.weight_GNN * gnn_temp + self.weight_INN * inn_temp
            temp_history.append[new_temp]
        return temp_history

    def print_loss(self, loss, t):
        print('-' * 40)
        print('Time: {}, MSE Loss: {}'.format(t, loss))
        print('-' * 40)

class GNNModel(torch.nn.Module):
    """GNN_Node
        This represents a room
    """

    def __init__(self, num_rooms):
        """Node initialization

        Args:
            temperature (_type_): _description_
        """
        super(GNNModel, self).__init__()

        self.adjacency_matrix = nn.Parameter(torch.ones(num_rooms, num_rooms) - torch.eye(num_rooms))
        # need vector for individual rooms
        self.self_temp_w = nn.Parameter(1)
        self.other_temp_w = nn.Parameter(1)
        self.temp = [0] * num_rooms

    def train_gnn(self, dataGNN, dataINN, y, INN, weight_gnn, weight_inn):
        '''
        Does one epoch of training for our GNN across the entire dataset

        :param dataGNN: data in format for GNN to process
        :param dataINN: data in format for INN to process
        :param y: correct temperature values
        :param INN: The current version of the INN model
        :param weight_gnn: The weight for our GNN
        :param weight_inn: The weight for our INN
        :return: The total loss over the dataset for this epoch
        '''
        loss_func = nn.MSELoss()
        loss_sum = 0
        for datum_idx in range(dataGNN.shape[0] - 1):
            current_temp = dataGNN[datum_idx]
            self.set_temps(current_temp)
            gnn_temp = self.predict(dataGNN[datum_idx])
            inn_temp = torch.zeros(dataGNN.shape[1])
            for room in range(dataGNN.shape[1]):
                inn_temp[room] = INN.predict(dataINN[datum_idx][room]).detach()
            # need weights in next line
            pred_y = self.temp + weight_gnn * (gnn_temp - self.temp) + weight_inn * (inn_temp - self.temp)
            loss = loss_func(pred_y, y)
            loss_sum += loss
            loss.backwards()
        return loss_sum / (len(dataGNN) - 1)

    def set_temps(self, temps):
        self.temp = temps

    def predict(self, data):
        aggregate_temp = torch.matmul(self.adjacency_matrix, data)
        final_temp = self.self_temp_w * self.temp + self.other_temp_w * aggregate_temp
        return final_temp


class INNModel(torch.nn.Module):
    """
    Room Network model:
        learning how the features of a room (AC, other stuff) affects the current temperature in a room
    Input training data:
    Output: Model
    """

    def __init__(self, dim_list) -> None:
        """
        Initialize the Room Network Model
        We'll change these inits later, specifics on the layers later.

        Args:
            features (list): Each entry in the list is a feature value of the room
        """
        super(INNModel, self).__init__()
        self.linear_list = nn.ModuleList([nn.Linear(dim_list[i], dim_list[i + 1]) for i in range(len(dim_list) - 1)])
        self.sigmoid = nn.Sigmoid()

    def forward(self, data, layer):
        return self.sigmoid(layer(data))

    def train_inn(self, dataGNN, dataINN, y, GNN, weight_gnn, weight_inn):
        '''
        Does one epoch of training for our INN across the entire dataset

        :param dataGNN: data in format for GNN to process
        :param dataINN: data in format for INN to process
        :param y: correct temperature values
        :param GNN: The current version of the GNN model
        :param weight_gnn: The weight for our GNN
        :param weight_inn: The weight for our INN
        :return: The total loss over the dataset for this epoch
        '''
        loss_func = nn.MSELoss()
        loss_sum = 0
        for datum_idx in range(len(dataINN)):
            time_idx = datum_idx % 2000  # filler number
            gnn_temp = GNN.predict(dataGNN[time_idx]).detach()
            inn_temp = self.predict(dataINN[datum_idx])
            cur_temp = dataINN[datum_idx][0]
            pred_y = cur_temp + weight_gnn * (gnn_temp - cur_temp) + weight_inn * (inn_temp - cur_temp)
            loss = loss_func(pred_y, y)
            loss_sum += loss
            loss.backwards()
        return loss_sum / len(dataINN)


    def predict(self, features):
        """
        Given the features, return some expected temperature
        """
        # room_temp = self.model.fit(features)
        # return room_temp
        values = features
        for layer_num in range(len(self.linear_list) - 1):
            values = self.forward(values, self.linear_list[layer_num])
        assert len(values) == 1, 'not proper INN output'
        return values




epochs = 100
stop_cond = 1e-3
dataGNN = 5
dataINN = 5
y = 5
num_rooms = dataGNN.shape[1]
network_layer_dims = [dataINN.shape[1], 10, 10, 1]
model = Model(num_rooms, network_layer_dims)
model.train(dataGNN, dataINN,y,epochs,stop_cond, visible=True)

