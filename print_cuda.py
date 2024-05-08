import torch
print('torch version: ',torch.__version__)
flag = torch.cuda.is_available()
if flag:
    print("CUDA is available")
else:
    print("CUDA NOT available")

ngpu= 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print("The driver is：",device)
print("GPU： ",torch.cuda.get_device_name(0))
