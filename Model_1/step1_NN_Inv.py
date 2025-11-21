import os
import re
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.init as init
import argparse
from model import UNet
from functions import tauc_tc_moment
from scipy.io import savemat

torch.manual_seed(3407)
np.random.seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)

def args_input():
    parse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument("--mode", type=str, default="Field_inv", help="Field_inv for real data inversion")
    parse.add_argument("--num_workers", type=int, default=4, help="Number of CPU to load data")
    parse.add_argument("--GPU", default="0", type=str, help="GPU number")
    parse.add_argument("--data_dir", type=str, help='path to real data directory')
    default_data_dir = './Target_events'
    parse.add_argument("--model_dir", type=str, default="./model/Model_1.pth")
    parse.add_argument("--batch_size", type=int, default=256)
    parse.add_argument("--output_dir", type=str, default="./results/1_NN_output_data", help='base path for output')
    parse.add_argument("--Fs", type=int, default=500, help='sampling frequency')
    args = parse.parse_args()
    if not args.data_dir:
        args.data_dir = default_data_dir
    return args

def gpu_set(args):
    if torch.cuda.is_available():
        cuda_kernel = args.GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_kernel
        print(f"CUDA_VISIBLE_DEVICES set to {cuda_kernel}")
        device = torch.device("cuda")
    else:
        print("No CUDA GPUs are available.")
        device = torch.device("cpu")
    return device

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)

class RealDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.filenames = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        print(f"Found {len(self.filenames)} files in {data_dir}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.filenames[idx])
        data = np.load(file_path, allow_pickle=True)

        input = torch.from_numpy(data['input']).float()
        toa = torch.from_numpy(data['toa']).float()
        az = torch.from_numpy(data['az']).float()
        sta = data['sta']  
        dist = data['tar_dist']
        
        evla = torch.tensor(data['evla']).float()
        evlo = torch.tensor(data['evlo']).float()
        evdp = torch.tensor(data['evdp']).float()
        stla = torch.from_numpy(data['stla']).float()
        stlo = torch.from_numpy(data['stlo']).float()
        
        stk = torch.tensor(data['stk']).float()
        dip = torch.tensor(data['dip']).float()
        rake = torch.tensor(data['rake']).float()
        
        basename = self.filenames[idx].replace('.npz', '')

        return input, toa, az, sta, dist, evla, evlo, evdp, stla, stlo, stk, dip, rake, basename

def custom_collate_fn(batch):
    input = torch.stack([item[0] for item in batch])
    toa = torch.stack([item[1] for item in batch])
    az = torch.stack([item[2] for item in batch])
    sta = [item[3] for item in batch]
    distances = [item[4] for item in batch]
    evla = torch.stack([item[5] for item in batch])
    evlo = torch.stack([item[6] for item in batch])
    evdp = torch.stack([item[7] for item in batch])
    stla = torch.stack([item[8] for item in batch])
    stlo = torch.stack([item[9] for item in batch])
    stk = torch.stack([item[10] for item in batch])
    dip = torch.stack([item[11] for item in batch])
    rake = torch.stack([item[12] for item in batch])
    basename = [item[13] for item in batch]
    
    return input, toa, az, sta, distances, evla, evlo, evdp, stla, stlo, stk, dip, rake, basename


def run_inversion(model, model_path, dataloader, device, output_folder, args):
    print(f"Loading model: {model_path}")
    checkpoint = torch.load(model_path, weights_only=True)
    if torch.cuda.device_count() > 1:
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    os.makedirs(output_folder, exist_ok=True)
    
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            (input, toa, az, sta, distances, evla, evlo, evdp, stla, stlo, stk, dip, rake, basename) = data
            print(f"Processing batch {i+1}/{len(dataloader)}")
            
            input_gpu = input.float().to(device)
            pred_gpu = model(input_gpu)

            pred_np = pred_gpu.cpu().numpy()
            input_np = input.numpy()
            toa_np = toa.numpy()
            az_np = az.numpy()
            distances = np.array(distances)
            
            data_num = input.shape[0]
            
            for b in range(min(args.batch_size, data_num)):

                pred_astfs = pred_np[b,:]
                tauc_pred_list = []
                tc_pred_list = []

                for row in range(pred_astfs.shape[1]):
                    pred_astfi = pred_astfs[:, row, :].reshape(-1)
                    tauc_pred, tc_pred = tauc_tc_moment(pred_astfi, 1/args.Fs)
                    tauc_pred_list.append(tauc_pred)
                    tc_pred_list.append(tc_pred)
                    
                Tauc_pred = np.array(tauc_pred_list)
                Tc_pred = np.array(tc_pred_list)

                save_path_mat = os.path.join(output_folder, f"{basename[b]}_inv.mat")
                savemat(save_path_mat, {
                    'input': input_np[b,:],
                    'pred': pred_np[b,:],
                    'az': az_np[b,:],
                    'toa': toa_np[b,:],
                    'distances': distances[b],
                    'sta': sta[b],
                    'stla': stla[b].numpy(),
                    'stlo': stlo[b].numpy(),
                    'evla': evla[b].item(),
                    'evlo': evlo[b].item(),
                    'evdp': evdp[b].item(),
                    'stk': stk[b].item(),
                    'dip': dip[b].item(),
                    'rake': rake[b].item(),
                    'Tauc_pred': Tauc_pred,
                    'Tc_pred': Tc_pred
                })
            print(f"Results saved to {output_folder}")

def invert_real_data(args, device):
    model_path = args.model_dir
    dataset = RealDataset(args.data_dir)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=custom_collate_fn
    )
    # print("Data successfully loaded")
    output_folder = args.output_dir
    model = UNet().to(device)
    initialize_weights(model)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    run_inversion(model, model_path, dataloader, device, output_folder, args)


if __name__ == '__main__':
    args = args_input()
    if args.mode == "Field_inv":
        print("-"*50 + "\nStart ASTFs extraction.\n")
        device = gpu_set(args)
        os.makedirs(args.output_dir, exist_ok=True)
        invert_real_data(args, device)
    else:
        print("Wrong input mode. Please use '--mode Field_inv'")