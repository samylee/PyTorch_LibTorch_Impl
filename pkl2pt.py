"""
This python script converts the network into Script Module---CPU
"""
import torch

# Download and load the pre-trained model
model = torch.load("model.pkl",map_location='cpu')

model.eval()

example_input = torch.rand(1, 3, 224, 224)
script_module = torch.jit.trace(model, example_input)
script_module.save('model_cpu.pt')

#"""
#This python script converts the network into Script Module---GPU
#"""
#import torch
#
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
## Download and load the pre-trained model
#model = torch.load("model.pkl")
#
#model.eval()
#
#example_input = torch.rand(1, 3, 224, 224)
#script_module = torch.jit.trace(model, example_input.to(device))
#script_module.save('model_gpu.pt')

