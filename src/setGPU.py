import torch
import copy

def gpu(model, dtype=torch.FloatTensor):
    if torch.cuda.is_available():
        gpu_dtype = torch.cuda.FloatTensor
        fixed_model_gpu = copy.deepcopy(model).type(gpu_dtype)
        return fixed_model_gpu, gpu_dtype
    return model, dtype

def save_model(state, filename='model.pth.tar'):
    torch.save(state.state_dict(), filename)

def load_model(filename='model.pth.tar'):    
    model = torch.load(filename)
    return model
