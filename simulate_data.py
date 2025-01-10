import os  
import numpy as np
from scipy.io import savemat
from scipy.ndimage import zoom
import nibabel as nib
import scipy.io as sio
from DCE_mat import Cosine8AIF_ExtKety

# Parameter ranges
Ktrans_range = [0.00001, 1]
Ve_range = [0.001, 1]
Vp_range = [0.001, 0.1]
dt_range = [0.5, 0.9]

# Load data
data = sio.loadmat('/DATA_Simu/BrainCancer.mat')
im = data['im']
data = sio.loadmat('/DATA_Simu/msak1.mat')
wt_mask3 = data['wt_mask3']
zoom_factors = (128 / 192, 128 / 192)

# Pre-calculate time points and AIF
num_scans = 35
time_per_point = 5.91 / 60
t = np.arange(0, num_scans * time_per_point, time_per_point)
aif = {'ab': 7.9785, 'ae': 0.5216, 'ar': 0.0482, 'mb': 32.8855, 'me': 0.1811, 'mm': 9.1868, 'mr': 15.8167, 't0': 0, 'tr': 0.2533}
aif['ab'] /= (1 - 0.45)

# Pre-generate masks for different unique intensities
magn_uni = np.unique(im)
nuni = len(magn_uni)
mask_uni = np.zeros((192, 192, nuni), dtype=bool)
for iu in range(nuni):
    mask_uni[:, :, iu] = im == magn_uni[iu]

# Noise levels to generate
noise_levels = [7, 25, 50, 100]

# Loop over each noise level and generate data
for noise_level in noise_levels:
    # Folder paths for train and test
    train_folder_path = f'/DATA_Simu/DATA_SD_{noise_level}/train/'
    test_folder_path = f'/DATA_Simu/DATA_SD_{noise_level}/test/'

    # Create directories if they don't exist
    os.makedirs(train_folder_path, exist_ok=True)
    os.makedirs(test_folder_path, exist_ok=True)

    # Generate 2000 images for the train folder and 400 images for the test folder
    for noise_idx in range(2000):
        # Define random parameters for the simulation
        Ktrans = np.concatenate(([0], np.random.rand(8) * (Ktrans_range[1] - Ktrans_range[0]) + Ktrans_range[0]))
        Ve = np.concatenate(([0], np.random.rand(8) * (Ve_range[1] - Ve_range[0]) + Ve_range[0]))
        Vp = np.concatenate(([0], np.random.rand(8) * (Vp_range[1] - Vp_range[0]) + Vp_range[0]))
        dt = np.concatenate(([0], np.random.rand(8) * (dt_range[1] - dt_range[0]) + dt_range[0]))

        # Generate parameter maps with masks and resize
        Ktrans_real = zoom(np.sum(mask_uni * np.expand_dims(Ktrans, axis=(0, 1)), axis=2), zoom_factors, order=1)
        ve_real = zoom(np.sum(mask_uni * np.expand_dims(Ve, axis=(0, 1)), axis=2), zoom_factors, order=1)
        vp_real = zoom(np.sum(mask_uni * np.expand_dims(Vp, axis=(0, 1)), axis=2), zoom_factors, order=1)
        dt_real = zoom(np.sum(mask_uni * np.expand_dims(dt, axis=(0, 1)), axis=2), zoom_factors, order=1)

        # Apply mask and remove NaN values
        Ktrans_real = np.nan_to_num(Ktrans_real * wt_mask3)
        ve_real = np.nan_to_num(ve_real * wt_mask3)
        vp_real = np.nan_to_num(vp_real * wt_mask3)
        dt_real = np.nan_to_num(dt_real * wt_mask3)

        # Vectorized DCE signal generation and noise addition
        vp_exp, ve_exp, ktrans_exp, dt_exp = vp_real[wt_mask3 > 0], ve_real[wt_mask3 > 0], Ktrans_real[wt_mask3 > 0], dt_real[wt_mask3 > 0]
        DCE_real = np.array([Cosine8AIF_ExtKety(t, aif, k, d, v, vp) for k, d, v, vp in zip(ktrans_exp, dt_exp, ve_exp, vp_exp)])
        DCE_real[np.isnan(DCE_real)] = 0
        S_noisy = np.where(DCE_real > 0.05, DCE_real + np.random.normal(0, 1 / noise_level, DCE_real.shape), DCE_real)
        S_noisy = np.maximum(S_noisy, 0)

        # Map back to full size arrays
        DCE_signal_real, DCE_signal_noisy = np.zeros((128, 128, 35)), np.zeros((128, 128, 35))
        DCE_signal_real[wt_mask3 > 0] = DCE_real
        DCE_signal_noisy[wt_mask3 > 0] = S_noisy

        # Concatenate signals with parameter maps for saving
        DCE_test = np.concatenate((DCE_signal_noisy, Ktrans_real[..., np.newaxis], ve_real[..., np.newaxis], vp_real[..., np.newaxis], dt_real[..., np.newaxis]), axis=2)
        DCE_test = np.transpose(DCE_test, (2, 0, 1))

        # Save the generated DCE data as a .nii file in the train folder
        filename_base = f'SD_{noise_level}_{noise_idx+1:05d}_Ct_params'
        new_filename = f'{filename_base}.nii'
        new_filename2 = os.path.join(train_folder_path, new_filename)
        img = nib.Nifti1Image(DCE_test, np.eye(4))
        nib.save(img, new_filename2)

        print(f'Saved {noise_idx+1}/2000 in train folder with noise level {noise_level:.2f}')

    # Generate 400 images for the test folder
    for noise_idx in range(2000):  # Index from 2000 to 2399
        # Define random parameters for the simulation (same as above)
        Ktrans = np.concatenate(([0], np.random.rand(8) * (Ktrans_range[1] - Ktrans_range[0]) + Ktrans_range[0]))
        Ve = np.concatenate(([0], np.random.rand(8) * (Ve_range[1] - Ve_range[0]) + Ve_range[0]))
        Vp = np.concatenate(([0], np.random.rand(8) * (Vp_range[1] - Vp_range[0]) + Vp_range[0]))
        dt = np.concatenate(([0], np.random.rand(8) * (dt_range[1] - dt_range[0]) + dt_range[0]))

        # Generate parameter maps with masks and resize
        Ktrans_real = zoom(np.sum(mask_uni * np.expand_dims(Ktrans, axis=(0, 1)), axis=2), zoom_factors, order=1)
        ve_real = zoom(np.sum(mask_uni * np.expand_dims(Ve, axis=(0, 1)), axis=2), zoom_factors, order=1)
        vp_real = zoom(np.sum(mask_uni * np.expand_dims(Vp, axis=(0, 1)), axis=2), zoom_factors, order=1)
        dt_real = zoom(np.sum(mask_uni * np.expand_dims(dt, axis=(0, 1)), axis=2), zoom_factors, order=1)

        # Apply mask and remove NaN values
        Ktrans_real = np.nan_to_num(Ktrans_real * wt_mask3)
        ve_real = np.nan_to_num(ve_real * wt_mask3)
        vp_real = np.nan_to_num(vp_real * wt_mask3)
        dt_real = np.nan_to_num(dt_real * wt_mask3)

        # Vectorized DCE signal generation and noise addition
        vp_exp, ve_exp, ktrans_exp, dt_exp = vp_real[wt_mask3 > 0], ve_real[wt_mask3 > 0], Ktrans_real[wt_mask3 > 0], dt_real[wt_mask3 > 0]
        DCE_real = np.array([Cosine8AIF_ExtKety(t, aif, k, d, v, vp) for k, d, v, vp in zip(ktrans_exp, dt_exp, ve_exp, vp_exp)])
        DCE_real[np.isnan(DCE_real)] = 0
        S_noisy = np.where(DCE_real > 0.05, DCE_real + np.random.normal(0, 1 / noise_level, DCE_real.shape), DCE_real)
        S_noisy = np.maximum(S_noisy, 0)

        # Map back to full size arrays
        DCE_signal_real, DCE_signal_noisy = np.zeros((128, 128, 35)), np.zeros((128, 128, 35))
        DCE_signal_real[wt_mask3 > 0] = DCE_real
        DCE_signal_noisy[wt_mask3 > 0] = S_noisy

        # Concatenate signals with parameter maps for saving
        DCE_test = np.concatenate((DCE_signal_noisy, Ktrans_real[..., np.newaxis], ve_real[..., np.newaxis], vp_real[..., np.newaxis], dt_real[..., np.newaxis]), axis=2)
        DCE_test = np.transpose(DCE_test, (2, 0, 1))

        # Save the generated DCE data as a .nii file in the test folder
        filename_base = f'SD_{noise_level}_{noise_idx+1:05d}_Ct_params'
        new_filename = f'{filename_base}.nii'
        new_filename2 = os.path.join(test_folder_path, new_filename)
        img = nib.Nifti1Image(DCE_test, np.eye(4))
        nib.save(img, new_filename2)

        print(f'Saved {noise_idx+1}/400 in test folder with noise level {noise_level:.2f}')
