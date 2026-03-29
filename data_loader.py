import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle
import time

# GPUMonitor to track GPU status
class GPUMonitor:
    """Monitors GPU usage and raises warnings"""
    def __init__(self, device):
        self.device = device
        self.is_cuda = device.type == 'cuda'
        self.last_check = time.time()
        self.warning_count = 0
        
        if self.is_cuda:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.has_pynvml = True
            except:
                self.has_pynvml = False
                print("⚠️  [WARNING] pynvml not available - install nvidia-ml-py3 for detailed GPU monitoring")
   
    def check_gpu_usage(self, context=""):
        """Check GPU utilization and warn if too low"""
        if not self.is_cuda:
            return
       
        now = time.time()
        if now - self.last_check < 5:  # Check every 5 seconds
            return
        self.last_check = now
       
        if self.has_pynvml:
            try:
                import pynvml
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                gpu_util = util.gpu
                mem_util = util.memory
               
                # Warn if GPU utilization is very low
                if gpu_util < 20:
                    self.warning_count += 1
                    print(f"\n⚠️  [GPU WARNING] Low GPU utilization: {gpu_util}% {context}")
                    print(f"    Expected: >60% during training")
                   
                if mem_util < 10:
                    print(f"⚠️  [GPU WARNING] Low VRAM usage: {mem_util}%")
                    print(f"    Expected: >30% with batch_size=512")
            except Exception as e:
                pass

# Define GPUDataset that loads data directly to GPU memory
class GPUDataset(Dataset):
    """Loads dataset directly to GPU memory for maximum speed"""
    def __init__(self, data_file, device, max_samples=None):
        global monitor
        print(f"[GPU] Loading dataset: {data_file}")
       
        # Check device
        if device.type != 'cuda':
            print("🚨 [CRITICAL] Dataset being loaded to CPU, not GPU!")
            print("    This will cause 10-15x slowdown!")
       
        # Load to CPU first
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
       
        if max_samples:
            data = data[:max_samples]
       
        print(f"[GPU] Transferring {len(data):,} samples to GPU...")
        start = time.time()
       
        # Transfer to GPU
        self.states = torch.stack([torch.FloatTensor(s) for s,_,_ in data]).to(device)
        self.policies = torch.stack([torch.FloatTensor(p) for _,p,_ in data]).to(device)
        self.values = torch.FloatTensor([[v] for _,_,v in data]).to(device)
       
        # Verify GPU placement
        if monitor:
            monitor.warn_if_cpu_tensors(self.states, "Dataset states")
            monitor.warn_if_cpu_tensors(self.policies, "Dataset policies")
            monitor.warn_if_cpu_tensors(self.values, "Dataset values")
       
        del data  # Free CPU memory
       
        elapsed = time.time() - start
        vram_gb = (self.states.element_size() * self.states.nelement() +
                   self.policies.element_size() * self.policies.nelement() +
                   self.values.element_size() * self.values.nelement()) / (1024**3)
       
        print(f"[GPU] ✓ Transfer complete in {elapsed:.1f}s")
        print(f"[GPU] Using {vram_gb:.2f} GB VRAM")
       
        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            print(f"[GPU] Total VRAM allocated: {allocated:.2f} GB")
   
    def __len__(self):
        return len(self.states)
   
    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]

# Multithreading setup for data loading
def get_data_loader(data_file, device, batch_size=1024, num_workers=16, max_samples=None):
    """Create a data loader with multithreading support"""
    # Create GPUDataset
    dataset = GPUDataset(data_file, device, max_samples=max_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return dataloader