import numpy as np
import math 
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FuncFormatter

def tauc_tc_moment(ASTF, dt):
    """
    This function is used to calculate tauc = 2 * sqrt(var(ASTF))
    ASTF: normalized Apparent Source Time Function
    dt: sample interval
    """
    # Reshape ASTF to a 1D array
    ASTF = ASTF.reshape(-1)
    # Generate time array
    t = np.arange(len(ASTF)) * dt
    # Calculate integral using the trapezoidal rule
    s = np.trapz(ASTF, dx=dt)
    if s == 0:
        print("Zero sum of ASTF")
        return 0, 0
    # Normalize ASTF
    ASTF = ASTF / s
    # Calculate tc
    tc = np.trapz(ASTF * t, dx=dt)
    # Calculate tauc
    tauc = 2 * np.sqrt(np.trapz(ASTF * (t-tc)**2, dx=dt))
    return tauc, tc


def normalize_waveform(waveform):
    max_val = np.max(np.abs(waveform))
    if max_val == 0:
        return waveform  # Return the waveform as is if the max is zero
    return waveform / max_val


def getDegree(latA, lonA, latB, lonB):
    """
    Args:
        point p1(latA, lonA)
        point p2(latB, lonB)
    Returns:
        The azimuth of B relative to A
        default: the basis of heading direction is north
    """
    radLatA = math.radians(latA)
    radLonA = math.radians(lonA)
    radLatB = math.radians(latB)
    radLonB = math.radians(lonB)
    dLon = radLonB - radLonA
    y = math.sin(dLon) * math.cos(radLatB)
    x = math.cos(radLatA) * math.sin(radLatB) - math.sin(radLatA) * math.cos(radLatB) * math.cos(dLon)
    brng = math.degrees(math.atan2(y, x))
    brng = (brng + 360) % 360
    return brng

def getDistance(latA, lonA, latB, lonB):
    ra = 6378140  # radius of equator: meter
    rb = 6356755  # radius of polar: meter
    flatten = (ra - rb) / ra  # Partial rate of the earth
    # change angle to radians
    radLatA = math.radians(latA)
    radLonA = math.radians(lonA)
    radLatB = math.radians(latB)
    radLonB = math.radians(lonB)
    pA = math.atan(rb / ra * math.tan(radLatA))
    pB = math.atan(rb / ra * math.tan(radLatB))
    x = math.acos(math.sin(pA) * math.sin(pB) + math.cos(pA) * math.cos(pB) * math.cos(radLonA - radLonB))
    c1 = (math.sin(x) - x) * (math.sin(pA) + math.sin(pB))**2 / math.cos(x / 2)**2
    c2 = (math.sin(x) + x) * (math.sin(pA) - math.sin(pB))**2 / math.sin(x / 2)**2
    dr = flatten / 8 * (c1 - c2)
    distance = ra * (x + dr)
    return distance

def get_snr(signal_window, noise_window):
    noise_rms = np.sqrt(np.mean(np.square(noise_window)))
    signal_rms = np.sqrt(np.mean(np.square(signal_window)))   
    snr = signal_rms / (noise_rms + 1e-10)
    return snr

def clean_astf(astf_data):
    cleaned_data = astf_data.copy()
    astf_2d = cleaned_data[0, :, :]
    for i in range(astf_2d.shape[0]):
        row = astf_2d[i, :]
        # Find first non-zero value
        first_non_zero = np.where(row != 0)[0]
        if len(first_non_zero) == 0:
            continue 
        first_non_zero_idx = first_non_zero[0]
        zero_indices = np.where(row[first_non_zero_idx:] == 0)[0]
        if len(zero_indices) > 0:
            zero_indices = zero_indices + first_non_zero_idx
            for j in range(len(zero_indices)-1):
                if zero_indices[j+1] == zero_indices[j] + 1:
                    zero_start = zero_indices[j]
                    astf_2d[i, zero_start:] = 0
                    break
    cleaned_data[0, :, :] = astf_2d
    return cleaned_data

def get_zero_positions(cleaned_astf):
    astf_2d = cleaned_astf[0, :, :]
    num_stations = astf_2d.shape[0]
    zero_positions = np.zeros(num_stations, dtype=np.int32)
    for i in range(num_stations):
        row = astf_2d[i, :]
        first_non_zero = np.where(row != 0)[0]
        if len(first_non_zero) == 0:
            zero_positions[i] = -1
            continue
        first_non_zero_idx = first_non_zero[0]
        zero_indices = np.where(row[first_non_zero_idx:] == 0)[0]
        if len(zero_indices) > 0:
            zero_positions[i] = zero_indices[0] + first_non_zero_idx
        else:
            zero_positions[i] = -1
    return zero_positions

def read_Tar(catalog_path, target_ids=None):
    train_data = open(catalog_path).read()
    train_events = [line.split() for line in train_data.strip().split('\n')]
    if target_ids is None:
        selected_events = train_events
    else:
        selected_events = [event for event in train_events if event[0] in [str(id) for id in target_ids]]
    if not selected_events:
        return None, None, None, None
    tar_id = str(selected_events[0][1])      # Target event ID
    tar_mags = float(selected_events[0][5])  # Target event magnitude
    egf_id = str(selected_events[0][6])      # eGf ID
    egf_mags = float(selected_events[0][10]) # eGf Magnitude
    return tar_id, tar_mags, egf_id, egf_mags

class SeismicWaveformPlotter_512:
    def __init__(self, real_target, pred_target_align, cc_maxs, misfits, s_egf, pred_astf, az, sta, match_ctlg):
        self.real_target = real_target
        self.pred_target = pred_target_align
        self.cc_maxs = cc_maxs
        self.misfits = misfits
        self.s_egf = s_egf
        self.astf = pred_astf
        self.az = az
        self.sta = sta
        self.match_ctlg = match_ctlg

    def normalize_waveform(self, waveform):
        max_val = np.max(np.abs(waveform))
        if max_val == 0:
            return waveform
        return waveform / max_val * 0.5

    def normalize_ASTF(self, waveform):
        max_val = np.max(np.abs(waveform))
        if max_val == 0:
            return waveform
        return waveform / max_val * 0.8

    def calc_tc_tauc(self, ASTF, dt):
        ASTF = ASTF.reshape(-1)
        t = np.arange(len(ASTF)) * dt
        s = np.trapz(ASTF, dx = dt)
        ASTF_norm = ASTF / s
        tc = np.trapz(ASTF_norm * t, dx = dt)
        tauc = 2 * np.sqrt(np.trapz(ASTF_norm * (t - tc)**2, dx = dt))
        return tc, tauc
    
    @staticmethod
    def format_func(x, p):
        if x == 0:
            return '0'
        else:
            return f'{x:.1f}'

    def plot_waveforms(self, save_path=None, snr=None, index=None, basename=None):
        # Convert cm to inches for figure size
        width_inches = 13 / 2.54   # 13 cm to inches
        height_inches = 20 / 2.54   # 9 cm to inches
        dt = 1/500  # sample interval
        
        fig = plt.figure(figsize=(width_inches, height_inches))
        gs = fig.add_gridspec(nrows=1, ncols=5, width_ratios=[1.5, 1.5, 0.6, 0.6, 0.8], wspace=0.1)
        axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
        y_limits = [-2.5, 68]
        nrows = self.astf.shape[1]

        # Target waves
        for j in range(nrows):
            t = np.arange(len(self.real_target[j, :])) * dt
            tar_waveform = self.normalize_waveform(self.real_target[j, :]) + j * 1
            axes[0].plot(t, tar_waveform, color='#737373', label='Field Data' if j == 0 else "", linestyle='--', linewidth=0.2)
            pred_tar = self.normalize_waveform(self.pred_target[j, :]) + j * 1
            axes[0].plot(t, pred_tar, color='#238443', label='Prediction' if j == 0 else "", linewidth=0.5)
        # eGf waves
        for j in range(nrows):
            t = np.arange(len(self.s_egf[j, :])) * dt
            egf_waveform = self.normalize_waveform(self.s_egf[j, :]) + j * 1
            axes[1].plot(t, egf_waveform, color='#737373', linestyle='-', linewidth=0.5)
        # CCs
        axes[2].axvline(x=0.895, color='#41ab5d', linestyle='--', linewidth=0.5)
        axes[2].axvline(x=0.795, color='#a1d99b', linestyle='--', linewidth=0.5)
        for idx, cc_value in enumerate(self.cc_maxs):
            if not np.isnan(cc_value):
                if cc_value >= 0.895:
                    axes[2].barh(idx, width=cc_value, color='#41ab5d', orientation="horizontal")
                elif cc_value < 0.895 and cc_value >= 0.795:
                    axes[2].barh(idx, width=cc_value, color='#a1d99b', orientation="horizontal")
                else:
                    axes[2].barh(idx, width=cc_value, color='#c7e9c0', orientation="horizontal")
        # Misfits
        axes[3].axvline(x=0.3, color='#bdbdbd', linestyle='--', linewidth=0.5)
        axes[3].axvline(x=0.5, color='#969696', linestyle='--', linewidth=0.5)
        for idx, misfit in enumerate(self.misfits):
            if not np.isnan(misfit):
                if misfit <= 0.25:
                    axes[3].barh(idx, width=misfit, color='#f0f0f0', orientation="horizontal")
                elif misfit > 0.25 and misfit <= 0.5:
                    axes[3].barh(idx, width=misfit, color='#bdbdbd', orientation="horizontal")
                else:
                    axes[3].barh(idx, width=misfit, color='#969696', orientation="horizontal")
                station_name = self.sta[idx]
                station_name = str(station_name.item())
                y_position = idx
                axes[3].text(0.62, y_position, f"{station_name}", 
                            verticalalignment='center', 
                            horizontalalignment='left', 
                            fontsize=4.2)
        
        # ASTFs
        cxlim_ASTF = [0, 0.16]
        xl, xr = cxlim_ASTF
        app_time_list = []
        cetroid_time_list = []
        for r in range(nrows):
            astfi = self.astf[:, r, :].reshape(-1)
            astfi[astfi < 0] = 0
            tauc_pred, tc_pred = self.calc_tc_tauc(astfi, dt)
            app_time_list.append(tauc_pred)
            cetroid_time_list.append(tc_pred)
        Tauc_S = np.array(app_time_list)
        Tc_S = np.array(cetroid_time_list)

        for p in range(nrows):
            # Center the ASTF
            astfi = self.astf[:, p, :].reshape(-1)
            t = np.arange(len(astfi)) * dt
            xm = round((xl + xr) / (2 * dt))
            if np.isnan(Tc_S[p]):
                xt = xm
            else:
                xt = round(Tc_S[p] / dt)
            shifted_stf = np.roll(astfi, xm - xt)
            shifted_stf = np.roll(astfi, 5)
            astfi_norm = self.normalize_ASTF(shifted_stf) + p * 1 -0.4
            axes[4].plot(t, astfi_norm, 'k', linewidth=0.4)
            axes[4].fill_between(t, p * 1 - 0.4, astfi_norm, where=(astfi_norm > p * 1 -0.4), color='#41ab5d')
            y = p * 1
            axes[4].text(0.145, y, f"{int(self.az[p]):.0f}°", verticalalignment='center', fontsize=4.2)
            axes[4].text(0.145, 66.3, f"Az", verticalalignment='center', fontsize=6)
        
        # Formatting subplots
        for ax in axes:
            ax.tick_params(axis='both', which='major', labelsize=6)
            ax.set_ylim(y_limits)

        axes[0].set_xlim(0, 512*dt)
        axes[0].set_ylabel('Stations', fontsize=8)
        axes[0].set_xlabel('Time (s)', fontsize=8)
        axes[0].set_title('Target Waveforms', fontsize=8, pad=5)
        axes[0].axvline(x=25*dt, color='gray', linestyle='--', linewidth=0.5)
        axes[0].tick_params(axis='both', labelsize=6, direction='in', which='both')
        axes[0].xaxis.set_major_formatter(FuncFormatter(self.format_func))
        axes[0].legend(loc='upper right', fontsize=5)
        
        axes[1].set_xlim(0, 512*dt)
        axes[1].set_xlabel('Time (s)', fontsize=8)
        axes[1].set_title('eGf Waveforms', fontsize=8, pad=5)
        axes[1].axvline(x=25*dt, color='gray', linestyle='--', linewidth=0.5)
        axes[1].tick_params(axis='both', labelsize=6, direction='in', which='both')
        axes[1].xaxis.set_major_formatter(FuncFormatter(self.format_func))
        # axes[1].legend(loc='upper right', fontsize=5)
        axes[1].set_yticks([])

        for ax in [axes[2], axes[3]]:
            ax.invert_yaxis()
            ax.set_xlabel('Coeff', fontsize=8)
            ax.get_yaxis().set_visible(False)
            ax.set_ylim(y_limits)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 1])
            ax.tick_params(axis='both', labelsize=6, direction='in', which='both')

        axes[4].set_xlim(cxlim_ASTF)
        axes[4].set_xlabel('Time (s)', fontsize=8)
        axes[4].set_xticks([0, 0.1, 0.2])
        axes[4].tick_params(axis='both', labelsize=6, direction='in', which='both')
        axes[4].set_title('ASTFs', fontsize=8, pad=5)
        axes[4].set_yticks([])
        axes[4].xaxis.set_major_formatter(FuncFormatter(self.format_func))

        axes[2].set_title('CCs', fontsize=8, pad=5)
        axes[3].set_title('Misfits', fontsize=8, pad=5)

        if basename:
            evt_num = basename.split('_')[0]
            evt_id = basename.split('_')[1]
            tar_id, tar_mags, egf_id, egf_mags = read_Tar(self.match_ctlg, target_ids=[evt_num])
            plt.suptitle(f"Results of Target {evt_id}, M = {tar_mags:.2f}\neGf {egf_id}, M = {egf_mags:.2f}", fontsize=10)
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, f'target_{evt_num}_{tar_mags}_{evt_id}_eGf_{egf_id}.png'), dpi=600, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
