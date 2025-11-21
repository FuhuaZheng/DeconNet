import os
import re
import numpy as np
from scipy.io import loadmat, savemat
from functions import *

# ------------------------------------------------------
results_folder = './results/'
catalog_folder = './catalog/'
evt_match_file = os.path.join(catalog_folder, 'Events_match.txt')
NN_outputs_data = os.path.join(results_folder, '1_NN_output_data')
ASTFs_plot_folder = os.path.join(results_folder, '2_pics_ASTFs')
filtered_data_folder = os.path.join(results_folder, '3_filtered_data')
# ------------------------------------------------------
NN_outputs = [f for f in os.listdir(NN_outputs_data) if f.endswith('inv.mat')]
os.makedirs(ASTFs_plot_folder, exist_ok=True)
os.makedirs(filtered_data_folder, exist_ok=True)

def extract_event_number(filename):
    match = re.search(r'target_(\d+)_*', filename)
    return int(match.group(1)) if match else float('inf')
NN_outputs_sorted = sorted(NN_outputs, key=extract_event_number)
cc_evts = []
print(NN_outputs_sorted)

for evt in NN_outputs:
    print(evt)
    evt_path = os.path.join(NN_outputs_data, evt)
    basename, extension = os.path.splitext(evt)
    evt_num = basename.split('_')[1]
    evt_id = basename.split('_')[2]
    egf_num = evt_num
    tar_id, tar_mags, egf_id, egf_mags = read_Tar(evt_match_file, target_ids=[evt_num])
    data = loadmat(evt_path)
    input = data['input']              # shape: (2, 64, 512)
    real_target = input[0,:,:]         # shape: (64, 512)
    s_egf = input[1,:,:]               # shape: (64, 512)
    pred_astf = data['pred']           # shape: (1, 64, 512)
    pred_astf = clean_astf(pred_astf)
    ASTFs_length = get_zero_positions(pred_astf)
    az = data['az']                    # shape: (64,)
    sta = data['sta'].astype(np.int32)                  # shape: (64,)
    # Extract other required data 
    stla = data.get('stla', None).squeeze()      # shape: (64,)
    stlo = data.get('stlo', None).squeeze()      # shape: (64,)
    evla = data.get('evla', None)
    evlo = data.get('evlo', None)
    evdp = data.get('evdp', None)
    stk = data.get('stk', None)
    dip = data.get('dip', None)
    rake = data.get('rake', None)
    toa = data.get('toa', None)
    distances = data.get('distances', None)
    Tauc_pred = data.get('Tauc_pred', None)
    Tc_pred = data.get('Tc_pred', None)
    # Calculate CC between real_target and pred_target 
    pred_target_not_align = np.zeros_like(real_target)      # shape: (64, 512)
    pred_target_align = np.zeros_like(real_target)          # shape: (64, 512)
    scaled_pred_target_align = np.zeros_like(real_target)   # shape: (64, 512)
    cc_maxs = np.zeros_like(az)                             # shape: (64,)
    misfits_512 = np.zeros_like(az)                         # shape: (64,)
    misfits_512_L1 = np.zeros_like(az)                      # shape: (64,)
    alphas = np.zeros_like(az)                              # shape: (64,)
    shifts = np.zeros_like(az)                              # shape: (64,)
    rows = real_target.shape[0]
    for row in range(rows):
        real_target_row = real_target[row, :]
        s_egf_row = s_egf[row, :]
        pred_astf_row = pred_astf[0, row, :]
        # Normalize the real target waveform
        real_target_row = normalize_waveform(real_target_row)
        # Keep full convolution result
        pred_target_full = np.convolve(s_egf_row, pred_astf_row, mode='full')
        # Use a temporary variable for normalization and correlation
        pred_target_temp = normalize_waveform(pred_target_full[:512])
        pred_target_temp = normalize_waveform(pred_target_temp)
        pred_target_not_align[row, :] = pred_target_temp
        # Calculate cross-correlation and find the best alignment
        real_demeaned = real_target_row - np.mean(real_target_row)
        pred_demeaned = pred_target_temp - np.mean(pred_target_temp)
        real_norm = np.sum(real_demeaned**2)
        pred_norm = np.sum(pred_demeaned**2)
        # Check if either signal has zero variance
        if real_norm < 1e-10 or pred_norm < 1e-10:
            # Cannot compute meaningful correlation, set to zero
            corr_function = np.zeros(1023)  # 2*512-1 = 1023
            max_corr = 0.0
        else:
            # Find shift using the temporary variable
            corr_function = np.correlate(real_demeaned, pred_demeaned, mode='full')
            # Normalize correlation coefficient
            corr_function /= np.sqrt(real_norm * pred_norm)
            # Find optimal shift
            shift = np.argmax(corr_function) - (len(corr_function) // 2)
            max_corr = np.max(corr_function)
        cc_maxs[row] = max_corr
        # Align the predicted target waveform (Apply shift)
        pred_target_shifted_full = np.roll(pred_target_full, shift)
        # Now truncate to 512 samples
        pred_target_row_aligned = pred_target_shifted_full[:512]          
        pred_target_align[row, :] = pred_target_row_aligned
        # Calculate optimal scaling factor alpha
        # alpha = (f_pre^T · f_obs) / (f_pre^T · f_pre)
        # Calculate optimal scaling factor alpha
        aligned_pred = pred_target_align[row, :]

        # Check if prediction contains valid data
        if np.all(aligned_pred == 0) or np.any(np.isnan(aligned_pred)):
            # If prediction is all zeros or has NaNs, set alpha to 0 and misfit to 1
            alpha = 0.0
            alphas[row] = alpha
            scaled_pred = np.zeros_like(aligned_pred)  # All zeros
            scaled_pred_target_align[row, :] = scaled_pred
            misfit_512 = 1.0  # Maximum misfit
        else:
            # Normal case - calculate optimal alpha
            dot_product = np.dot(aligned_pred, real_target_row)
            norm_squared = np.dot(aligned_pred, aligned_pred)
            
            # Additional safety check for very small denominator
            if norm_squared < 1e-10:
                alpha = 0.0
            else:
                alpha = dot_product / norm_squared
            
            alphas[row] = alpha
            
            # Scale the aligned predicted waveform by alpha
            scaled_pred = alpha * aligned_pred
            scaled_pred_target_align[row, :] = scaled_pred
            
            # Calculate misfit with the scaled prediction
            # Start point has already been aligned
            misfit_512 = np.sum((real_target_row - scaled_pred)**2) / np.sum(real_target_row**2)
            misfit_512_L1 = np.sum(np.abs(real_target_row - scaled_pred)) / np.sum(np.abs(real_target_row))

        misfits_512[row] = misfit_512
        misfits_512_L1[row] = misfit_512_L1
        shifts[row] = shift
    # Filter rows with misfit < 0.3
    good_indices_512_03 = np.where(misfits_512 < 0.3)[0]
    
    if len(good_indices_512_03) > 0:
        # Extract ASTF for good indices
        astf_nn = pred_astf[0]  # shape: (64, 512)
        # 512 misfits
        astf_nn_misfit_512 = astf_nn[good_indices_512_03]
        sta_misfit_512 = sta[good_indices_512_03] if sta is not None else None
        stla_misfit_512 = stla[good_indices_512_03] if stla is not None else None
        stlo_misfit_512 = stlo[good_indices_512_03] if stlo is not None else None
        # Create filtered data dictionary
        filtered_data = {
            'real_tar_S': real_target,                        # shape: (64, 512)
            'pred_tar_S': pred_target_align,                  # shape: (64, 512)
            'scaled_pred_tar_S': scaled_pred_target_align,    # shape: (64, 512)
            'CCs_S': cc_maxs,                                 # shape: (64,)
            'misfits_512': misfits_512,                       # shape: (64,)
            'misfits_512_L1': misfits_512_L1,                 # shape: (64,)
            'alphas_S': alphas,                               # shape: (64,)
            's_egf': s_egf,                                   # shape: (64, 512)
            'astf_nn_S': astf_nn,                             # shape: (64, 512)
            'target_num_S': evt_num,
            'target_id_S': evt_id,
            'az_S': az,                                       # shape: (64,)
            'toa_S': toa,                                     # shape: (64,)
            'distances': distances,                           # shape: (64,)
            'Tauc': Tauc_pred,                                # shape: (64,)
            'Tc': Tc_pred,                                    # shape: (64,)
            'ASTFs_length': ASTFs_length,                     # shape: (64,)
            'sta_S': sta,                                     # shape: (64,)
            'stla_S': stla,                                   # shape: (64,)
            'stlo_S': stlo,                                   # shape: (64,)
            ############################
            'astf_nn_misfit_512': astf_nn_misfit_512,         # shape: (n, 512)
            'sta_misfit_512': sta_misfit_512,                 # shape: (n,)
            'stla_misfit_512': stla_misfit_512,               # shape: (n,)
            'stlo_misfit_512': stlo_misfit_512,               # shape: (n,)
            'indices_03_512': good_indices_512_03,            # shape: (n,)
            'misfits_filtered_512': misfits_512[good_indices_512_03],  # shape: (n,)
        }
        # Add event information (unchanged)
        if evla is not None: filtered_data['evla'] = evla
        if evlo is not None: filtered_data['evlo'] = evlo
        if evdp is not None: filtered_data['evdp'] = evdp
        if stk is not None: filtered_data['stk'] = stk
        if dip is not None: filtered_data['dip'] = dip
        if rake is not None: filtered_data['rake'] = rake
        # Save filtered data
        filtered_mat_path = os.path.join(filtered_data_folder, f"target_{evt_num}_{evt_id}_eGf{egf_num}_{egf_id}.mat")
        savemat(filtered_mat_path, filtered_data)

    # Calculate mean CC for all stations
    cc_maxs = np.nan_to_num(cc_maxs, nan=0)
    cc_evt = np.mean(cc_maxs)
    cc_evts.append(cc_evt)

    plot_512 = SeismicWaveformPlotter_512(real_target, scaled_pred_target_align, cc_maxs, misfits_512, s_egf, pred_astf, az, sta, evt_match_file)
    plot_512.plot_waveforms(ASTFs_plot_folder, basename=f'{evt_num}_{evt_id}')