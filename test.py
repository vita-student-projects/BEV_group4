import torch

num_gpus = torch.cuda.device_count()
print(num_gpus)