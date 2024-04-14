import util
import torch
from model import Net

device = util.get_device() 
path = "task_1_new_model.pth"    # take model from train.py
model = Net().to(device)
model.load_state_dict(torch.load(path))
util.get_test_set_preformance(model, device)
