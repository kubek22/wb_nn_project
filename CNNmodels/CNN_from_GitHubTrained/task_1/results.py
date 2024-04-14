import util
import torch
from model import Net

device = util.get_device() 
path = "task_1_my_model.pth"    
model = Net().to(device)
model.load_state_dict(torch.load(path))
util.get_test_set_preformance(model, device)
