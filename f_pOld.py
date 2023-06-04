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

    def __init__(self, num_rooms, INNdim_list, room_order, loss_func = nn.MSELoss()) -> None:
        """Init

        Args:
            num_rooms (int): Amount of rooms
            INNdim_list (list of int): dimensions for each layer of individual network

            n amount of rooms x n amount of rooms 

        """
        super(Model, self).__init__()

        self.GNN = GNNModel(num_rooms, loss_func, room_order)

        self.INN = INNModel(INNdim_list, loss_func)

        self.weight_INN = nn.Parameter(torch.ones(1))
        self.weight_GNN = nn.Parameter(torch.ones(1))

    def train(self, dataGNN, dataINN, y, epochs, epsilon, max_runtime, visible = False):
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
        total_start = time.time()

        gnn_improv = 1
        inn_improv = 1
        stop_next = False

        plt.figure(figsize=(12, 6))
        plt.ion()
        plt.show()
        while True:
            plt.clf()

            epoch_start = time.time()
            if choice != 1:
                lossGNN = self.GNN.train_gnn(torch.clone(dataGNN), torch.clone(dataINN), self.INN, self.weight_GNN, self.weight_INN)
                if t >= 2:
                    gnn_improv = lossGNN / loss_record_gnn[-1]
                gnn_amount += 1
                t += 1
            if choice != 0:
                lossINN = self.INN.train_inn(torch.clone(dataGNN), torch.clone(dataINN), y, self.GNN, self.weight_GNN, self.weight_INN)
                if t >= 2:
                    inn_improv = lossINN / loss_record_inn[-1]
                inn_amount += 1
                t += 1
            loss_total = lossGNN + lossINN
            loss_record_total.append(loss_total)
            loss_record_gnn.append(lossGNN)
            loss_record_inn.append(lossINN)

            if t >= 3:
                choice = 0 if gnn_improv < inn_improv else 1

            if visible:
                self.print_loss(loss_total, t, epoch_start, gnn_improv, inn_improv)


            plt.plot(range(len(loss_record_gnn)), loss_record_gnn, label= 'GNN Loss: {}'.format(gnn_amount))
            plt.plot(range(len(loss_record_inn)), loss_record_inn, label= 'INN Loss: {}'.format(inn_amount))
            plt.plot(range(len(loss_record_total)), loss_record_total, label= 'Total Loss: {}'.format(t))
            plt.yscale('log')
            plt.legend()

            plt.draw()
            plt.pause(.001)
            plt.savefig('images\\loss_graph.png')

            if 2 - (inn_improv + gnn_improv) < epsilon or (t > epochs and epochs != 0):
                choice = 2
                if stop_next:
                    break
                stop_next = True
            elif time.time() - total_start >= max_runtime:
                break
            else:
                stop_next = False


        torch.save(model.state_dict(), 'runtime data\\model_save.pt')
        np.savez('runtime data\\loss records.npz', loss_record_gnn, loss_record_inn, loss_record_total)
        print('Model saved')
        print('Total time taken: {}'.format(time.time() - total_start))

    def predict(self, temp_start, features):
        dims = features.shape
        gnn_temp = self.GNN.predict(temp_start).detach()
        inn_temp = torch.zeros(dims[0])
        for room in range(dims[0]):
            inn_temp[room] = self.INN.predict(features[room]).detach()
        new_temp = temp_start + self.weight_GNN * (gnn_temp - temp_start) + self.weight_INN * (inn_temp - temp_start )
        return new_temp

    def print_loss(self, loss, t, start, gnn_improvement, inn_improvement):
        print('-' * 65)
        print('Epochs: {}, Time: {}, MSE Loss: {}'.format(t, time.time()-start, loss))
        print('GNN Improvement: {}, INN improvement: {}'.format(gnn_improvement, inn_improvement))
        print('-' * 65)

class GNNModel(torch.nn.Module):
    """GNN_Node
        This represents a room
    """

    def __init__(self, num_rooms, loss_func, room_order):
        """Node initialization

        Args:
            temperature (_type_): _description_
        """
        super(GNNModel, self).__init__()



        self.adjacency_matrix = nn.Parameter(self.create_matrix(num_rooms,room_order))
        # need vector for individual rooms

        self.self_temp_w = nn.Parameter(torch.ones(1)*.5)
        self.other_temp_w = nn.Parameter(torch.ones(1)*.5)
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
        matrix = (torch.ones(num_rooms, num_rooms) - torch.eye(num_rooms))/num_rooms
        return matrix


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
            gnn_temp = self.predict(torch.clone(dataGNN[datum_idx]))
            inn_temp = torch.zeros(dataGNN.shape[1]).to(gnn_temp.device)
            for room in range(dataGNN.shape[1]):
                inn_temp[room] = INN.predict(dataINN[room][datum_idx]).detach()
            # need weights in next line
            pred_y = current_temp + weight_gnn * (gnn_temp - current_temp) + weight_inn * (inn_temp - current_temp)
            y = dataGNN[datum_idx+1]
            loss = self.loss_func(pred_y, y)
            loss_sum += float(loss)
            loss.backward()
            optimizer.step()
            if datum_idx == 0:
                print('GNN Datum Idx:')
            if datum_idx % 100 == 0:
                 print(datum_idx, end= ' ')
            # if datum_idx > 100:
            #     break

        print('')
        self.ensure_self_zero()
        return loss_sum

    def ensure_self_zero(self):
        optimizer.zero_grad()
        size = self.adjacency_matrix.shape[0]
        loss_func_zero = nn.MSELoss()
        loss = 10 * loss_func_zero(torch.zeros(size).to(self.adjacency_matrix.device), self.relu(torch.diagonal(self.adjacency_matrix)))
        loss.backward()
        optimizer.step()

    def predict(self, data):
        aggregate_temp = torch.matmul(self.adjacency_matrix, data)
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

        for time_idx in range(dataINN.shape[1]-1):
            gnn_temp = GNN.predict(torch.clone(dataGNN[time_idx])).detach()
            for room_idx in range(dataINN.shape[0]):
                optimizer.zero_grad()
                inn_temp = self.predict(torch.clone(dataINN[room_idx][time_idx]))
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
            # if time_idx > 100:
            #     break


        print('')
        return loss_sum / (dataINN.shape[0])


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



time_interval = 120# Data

dataINN = torch.from_numpy(np.load(f'preprocessing_output\\merged_features_rooms_{str(time_interval)}T.npy').astype(np.float32))
yINN = torch.from_numpy(np.load(f'preprocessing_output\\merged_features_rooms_{str(time_interval)}T.npy').astype(np.float32)[:,:,1])
dataGNN = torch.from_numpy(np.load(f'preprocessing_output\\merged_temps_time_{str(time_interval)}T.npy').astype(np.float32))
rooms = np.load(f'preprocessing_output\\merged_rooms_list.npy')

# Device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
    dataINN = dataINN.cuda()
    yINN = yINN.cuda()
    dataGNN = dataGNN.cuda()
else:
    device = torch.device("cpu")


# Hyperparameters
epochs = 15
stop_epsilon = 0.01
max_runtime = 36000
num_rooms = dataINN.shape[0]
network_layer_dims = [dataINN.shape[2], 10, 10, 1]
loss_func = nn.L1Loss()
learning_rate = .01


if __name__ == '__main__':
    model = Model(num_rooms, network_layer_dims, rooms, loss_func)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train(dataGNN, dataINN, yINN, epochs,stop_epsilon, max_runtime, visible=True)

