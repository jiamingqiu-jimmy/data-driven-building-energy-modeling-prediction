import torch
import torch.nn as nn
import numpy as np
import f_p




def simulate_single_step(model, base_temp, features):
    new_temp = model.predict(base_temp, features).detach()
    return new_temp

def simulate_all(model, data):
    temp = data[:,0,1]
    dims = data.shape
    for datum in range(dims[1]):
        features = data[:,datum,:]
        features[:,1] = temp
        temp = simulate_single_step(model, temp, features)
    return temp

def check_final_temp(temp, data):
    final_temp_true = data[:,-1,1].numpy()
    temp = temp.numpy()
    return np.linalg.norm(temp-final_temp_true)


model = f_p.Model(81, [19,10,10,1])
model.load_state_dict(torch.load('runtime data\\model_save.pt'))
model = model.to(torch.device('cpu'))
data = torch.from_numpy(np.load('preprocessing_output\\merged_features_rooms.npy').astype(np.float32)[:,0:1129,:])
final_temp = simulate_all(model, data)
loss = check_final_temp(final_temp, data)
print(loss)