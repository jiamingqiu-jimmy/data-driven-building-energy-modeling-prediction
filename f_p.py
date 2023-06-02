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


# from torch_geometric.nn import GCNConv
# from torch_geometric.datasets import Planetoid


# In[6]:


class Model(nn.Module):
    """ 
    Graph Network model connecting all the rooms in a building
        Learns how a heat radiates between rooms, given each room's temperature.
    """

    def __init__(self, num_rooms, INNdim_list, loss_func = nn.MSELoss()) -> None:
        """Init

        Args:
            num_rooms (int): Amount of rooms
            INNdim_list (list of int): dimensions for each layer of individual network

            n amount of rooms x n amount of rooms 

        """
        super(Model, self).__init__()

        self.GNN = GNNModel(num_rooms, loss_func)

        self.INN = INNModel(INNdim_list, loss_func)

        self.weight_INN = nn.Parameter(torch.ones(1))#.cuda()
        self.weight_GNN = nn.Parameter(torch.ones(1))#.cuda()

    def train(self, dataGNN, dataINN, y, epochs, epsilon, visible = False):
        """
        Loss Function: Actual temp - Predicted Temp
        Predicted Temp = af + bg

        Returns:
            _type_: _description_
        """

        # data prep
        loss_record_total = []
        loss_record_gnn = []
        loss_record_inn = []
        lossGNN = 0
        lossINN = 0

        t = 0
        gnn_amount = 0
        inn_amount = 0
        choice = 2

        gnn_improv = 1
        inn_improv = 1

        plt.figure(figsize=(12, 6))
        plt.ion()
        plt.show()
        while True:
            plt.clf()

            start = time.time()
            if choice != 1:
                lossGNN = self.GNN.train_gnn(dataGNN, dataINN, self.INN, self.weight_GNN, self.weight_INN)
                gnn_amount += 1
                if t >= 1:
                    gnn_improv = 1 - lossGNN / loss_record_gnn[-1]
            if choice != 0:
                lossINN = self.INN.train_inn(dataGNN, dataINN, y, self.GNN, self.weight_GNN, self.weight_INN)
                inn_amount += 1
                if t >= 1:
                    inn_improv = 1 - lossINN / loss_record_inn[-1]
            loss_total = lossGNN + lossINN
            loss_record_total.append(loss_total)
            loss_record_gnn.append(lossGNN)
            loss_record_inn.append(lossINN)
            t += 1
            if t >= 2:
                print(gnn_improv, inn_improv)
                choice = 0 if gnn_improv > inn_improv else 1

            if visible:
                self.print_loss(loss_total, t, start)

            if inn_improv + gnn_improv < epsilon or (t > epochs and t != 0):
                break

            plt.plot(range(len(loss_record_gnn)), loss_record_gnn, label= 'GNN Loss: {}'.format(gnn_amount))
            plt.plot(range(len(loss_record_inn)), loss_record_inn, label= 'INN Loss: {}'.format(inn_amount))
            plt.plot(range(len(loss_record_total)), loss_record_total, label= 'Total Loss: {}'.format(t+2))
            plt.yscale('log')
            plt.legend()

            plt.draw()
            plt.pause(.001)
            plt.savefig('images\\loss_graph.png')
        torch.save(model.state_dict(), 'runtime data\\model_save.pt')
        np.savez('runtime data\\loss records.npz', loss_record_gnn, loss_record_inn, loss_record_total)
        print('Model saved')

    def predict(self, temp_start, features):
        dims = features.shape
        gnn_temp = self.GNN.predict(temp_start).detach()
        inn_temp = torch.zeros(dims[0])
        for room in range(dims[0]):
            inn_temp[room] = self.INN.predict(features[room]).detach()
        new_temp = temp_start + self.weight_GNN * (temp_start - gnn_temp) + self.weight_INN * (temp_start - inn_temp)
        return new_temp

    def print_loss(self, loss, t, start):
        print('-' * 65)
        print('Epoch: {}, Time: {}, MSE Loss: {}'.format(t, time.time()-start, loss))
        print('-' * 65)

class GNNModel(torch.nn.Module):
    """GNN_Node
        This represents a room
    """

    def __init__(self, num_rooms, loss_func):
        """Node initialization

        Args:
            temperature (_type_): _description_
        """
        super(GNNModel, self).__init__()

        self.adjacency_matrix = nn.Parameter(torch.ones(num_rooms, num_rooms) - torch.eye(num_rooms))#.cuda()
        # need vector for individual rooms

        self.self_temp_w = nn.Parameter(torch.ones(1))
        self.other_temp_w = nn.Parameter(torch.ones(1))
        self.temp = torch.zeros(num_rooms)
        self.loss_func = loss_func

    def train_gnn(self, dataGNN, dataINN, INN, weight_gnn, weight_inn):
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

        loss_sum = 0
        for datum_idx in range(dataGNN.shape[0] - 1):
            optimizer.zero_grad()
            current_temp = dataGNN[datum_idx]
            self.set_temps(current_temp)
            gnn_temp = self.predict(dataGNN[datum_idx])
            inn_temp = torch.zeros(dataGNN.shape[1]).to(gnn_temp.device)
            for room in range(dataGNN.shape[1]):
                inn_temp[room] = INN.predict(dataINN[room][datum_idx]).detach()
            # need weights in next line
            pred_y = self.temp + weight_gnn * (gnn_temp - self.temp) + weight_inn * (inn_temp - self.temp)
            y = dataGNN[datum_idx+1]
            loss = self.loss_func(pred_y, y)
            loss_sum += float(loss)
            loss.backward()
            optimizer.step()
            if datum_idx == 0:
                print('GNN Datum Idx:')
            if datum_idx % 100 == 0:
                 print(datum_idx, end= ' ')
            if datum_idx > 10:
                break

        print('')
        return loss_sum #/ (len(dataGNN) - 1)

    def set_temps(self, temps):
        self.temp = temps.to(temps.device)

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

    def __init__(self, dim_list, loss_func) -> None:
        """
        Initialize the Room Network Model
        We'll change these inits later, specifics on the layers later.

        Args:
            features (list): Each entry in the list is a feature value of the room
        """
        super(INNModel, self).__init__()
        self.linear_list = nn.ModuleList([nn.Linear(dim_list[i], dim_list[i + 1]) for i in range(len(dim_list) - 1)])
        # self.sigmoid = nn.Sigmoid()
        self.loss_func = loss_func


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
        loss_sum = 0

        for time_idx in range(dataINN.shape[1]):
            gnn_temp = GNN.predict(dataGNN[time_idx]).detach()
            for room_idx in range(dataINN.shape[0]):
                optimizer.zero_grad()
                inn_temp = self.predict(dataINN[room_idx][time_idx])
                cur_temp = dataINN[room_idx][time_idx][1]
                pred_y = cur_temp + weight_gnn * (gnn_temp[room_idx] - cur_temp) + weight_inn * (inn_temp - cur_temp)
                loss = loss_func(pred_y, y[room_idx][time_idx+1].unsqueeze(0))
                loss_sum += float(loss)
                loss.backward()
                optimizer.step()


            if time_idx == 0:
                print('INN Datum Idx:')
            if time_idx % 100 == 0:
                 print(time_idx, end= ' ')
            if time_idx > 10:
                break


        print('')
        return loss_sum #/ len(dataINN)


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
dataINN = torch.from_numpy(np.load('preprocessing_output\\merged_features_rooms.npy').astype(np.float32)[:,0:1129,:])
yINN = torch.from_numpy(np.load('preprocessing_output\\merged_features_rooms.npy').astype(np.float32)[:,0:1130,1])
dataGNN = torch.from_numpy(np.load('preprocessing_output\\merged_temps_time.npy').astype(np.float32)[0:1130])

# Device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
    dataINN = dataINN.cuda()
    yINN = yINN.cuda()
    dataGNN = dataGNN.cuda()
else:
    device = torch.device("cpu")


# Hyperparameters
epochs = 1000
stop_epsilon = .001
num_rooms = dataINN.shape[0]
network_layer_dims = [dataINN.shape[2], 10, 10, 1]
loss_func = nn.MSELoss()
learning_rate = .01

if __name__ == '__main__':
    model = Model(num_rooms, network_layer_dims, loss_func)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train(dataGNN, dataINN, yINN, epochs,stop_epsilon, visible=True)

