#!/usr/bin/env python3
import platform
import torch

# NB diffvg does not seem to like M1 (segfaults etc)
if False: #platform.processor() == 'arm': # and not settings.no_arm:
    print("Detected Mac M1/M2")
    device_name = 'mps'
    has_gpu = False
    torch.set_default_tensor_type(torch.FloatTensor)
    diffvg_device_name = 'cpu'
    torch_dtype = torch.float32
else:
    if torch.cuda.is_available():
        device_name = 'cuda'
        diffvg_device_name = 'cuda'
        has_gpu = True
        torch_dtype = torch.float32
    else:
        device_name = 'cpu'
        diffvg_device_name = 'cpu'
        has_gpu = False
        torch_dtype = torch.float32

print('Setting device to: ', device_name)
device = torch.device(device_name)
diffvg_device = torch.device(diffvg_device_name)

def clear_memory():
    if has_gpu:
        import gc
        torch.cuda.empty_cache()
        gc.collect()
