import torch
import torch.nn as nn
import numpy as np
import f_pOld
import f_pNew




def simulate_single_step(model, base_temp, features):
    new_temp = model.predict(base_temp, features).detach()
    return new_temp

def simulate_all(model, data):
    temp = data[0,:,1]
    dims = data.shape
    for datum in range(dims[1]):
        features = torch.clone(data[datum,:,:])
        features[:,1] = torch.clone(temp)
        temp = simulate_single_step(model, temp, features)
    return temp

def check_final_temp(temp, data):
    final_temp_true = data[-1,:,1].numpy()
    temp = temp.numpy()
    return np.linalg.norm(temp-final_temp_true)


if __name__ == '__main__':
    rooms = np.load(f'preprocessing_output\\merged_rooms_list.npy')
    model = f_pNew.Model(80, [19, 20, 20, 1], rooms)
    time_interval = 30
    model.load_state_dict(torch.load('runtime data\\model2_save.pt'))
    model = model.to(torch.device('cpu'))
    data = torch.from_numpy(np.load('preprocessing_output/120T/features_rooms_testing_120T.npy').astype(np.float32))
    final_temp = simulate_all(model, torch.clone(data))
    loss = check_final_temp(final_temp, torch.clone(data))

    final_adj = model.GNN.adjacency_matrix.detach().numpy()
    data_end = data.numpy()[:,-1,1]
    final_temp = final_temp.numpy()
    print(loss)
    data_test = torch.from_numpy(np.load('preprocessing_output/120T/features_rooms_testing_120T.npy').astype(np.float32))
    error = 0
    for datum_idx in range(data_test.shape[0]-1):
        start_temp = data_test[datum_idx, :, 1]
        features = data_test[datum_idx,:,:]
        pred_y = simulate_single_step(model, start_temp, features)

        loss_func = nn.MSELoss()
        true_y = data_test[datum_idx+1, :, 1]
        error += float(loss_func(pred_y, true_y).detach())
    print(error)
