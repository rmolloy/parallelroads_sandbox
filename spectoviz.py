import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

# Constants and helper functions we defined earlier
epsilon = 1e-10

def compute_fft_spectrum(audio_data, sample_rate):
    num_samples = len(audio_data)
    frequencies = np.fft.rfftfreq(num_samples, d=1/sample_rate)
    magnitudes = np.abs(np.fft.rfft(audio_data))
    return frequencies, magnitudes / num_samples

def average_within_log_bins(frequencies, magnitudes):
    bin_edges = np.geomspace(20, 20000, num=50)
    bin_indices = np.digitize(frequencies, bin_edges)
    binned_frequencies = (bin_edges[:-1] + bin_edges[1:]) / 2
    binned_magnitudes = [np.mean(magnitudes[bin_indices == i]) for i in range(1, len(bin_edges))]
    return binned_frequencies, binned_magnitudes

def equal_loudness_correction(frequencies):
    iso_226_freqs = np.array([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000])
    iso_226_phon_80dB = np.array([64.3, 58.3, 53.3, 49.6, 47.0, 44.7, 43.0, 41.7, 40.7, 40.0, 39.4, 39.0, 38.7, 38.5, 38.4, 38.4, 38.5, 38.6, 38.9, 39.4, 40.0, 40.7, 41.5, 42.4, 43.4, 44.3, 45.6, 47.0, 48.3, 49.7, 80.0])
    return np.interp(frequencies, iso_226_freqs, iso_226_phon_80dB)

# Normalization function
def normalize_audio_files(wav_file_paths):
    max_amplitude = 0
    for wav_file_path in wav_file_paths:
        _, audio_data = wavfile.read(wav_file_path)
        max_amplitude = max(max_amplitude, np.max(np.abs(audio_data)))
    normalized_audios = []
    for wav_file_path in wav_file_paths:
        _, audio_data = wavfile.read(wav_file_path)
        normalized_audio = (audio_data / np.max(np.abs(audio_data))) * max_amplitude
        normalized_audios.append(normalized_audio)
    return normalized_audios

# Visualization function for normalized audio files
def plot_normalized_fft_average_spectra_log_binned_line_corrected(wav_file_paths):
    normalized_audios = normalize_audio_files(wav_file_paths)
    plt.figure(figsize=(10, 6))
    for idx, normalized_audio in enumerate(normalized_audios):
        sample_rate, _ = wavfile.read(wav_file_paths[idx])
        if len(normalized_audio.shape) == 2:
            normalized_audio = np.mean(normalized_audio, axis=1)
        frequencies, magnitude = compute_fft_spectrum(normalized_audio, sample_rate)
        mask = (frequencies >= 20) & (frequencies <= 20000)
        frequencies = frequencies[mask]
        magnitude = magnitude[mask]
        binned_frequencies, binned_magnitudes = average_within_log_bins(frequencies, magnitude)
        correction = equal_loudness_correction(binned_frequencies)
        corrected_magnitudes = 10 * np.log10(np.array(binned_magnitudes) + epsilon) - correction
        plt.plot(binned_frequencies, corrected_magnitudes, label=os.path.basename(wav_file_paths[idx]))
    plt.xscale('log')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    plt.title('Corrected FFT-based Logarithmically Binned Average Spectra for Human Hearing Range (Normalized Volumes)')
    plt.legend()
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.tight_layout()
    plt.show()

def plot_normalized_fft_average_spectra_log_binned_line_corrected_adjusted_min(wav_file_paths):
    normalized_audios = normalize_audio_files(wav_file_paths)
    plt.figure(figsize=(10, 6))
    
    for idx, normalized_audio in enumerate(normalized_audios):
        sample_rate, _ = wavfile.read(wav_file_paths[idx])
        
        # If stereo, average the channels
        if len(normalized_audio.shape) == 2:
            normalized_audio = np.mean(normalized_audio, axis=1)
        
        # Compute the FFT-based magnitude spectrum
        frequencies, magnitude = compute_fft_spectrum(normalized_audio, sample_rate)
        
        # Filter for the typical range of human hearing (20Hz to 20kHz)
        mask = (frequencies >= 20) & (frequencies <= 20000)
        frequencies = frequencies[mask]
        magnitude = magnitude[mask]
        
        # Average the magnitudes within logarithmically spaced frequency bins
        binned_frequencies, binned_magnitudes = average_within_log_bins(frequencies, magnitude)
        
        # Correct the magnitudes based on equal loudness contours
        correction = equal_loudness_correction(binned_frequencies)
        corrected_magnitudes = 10 * np.log10(np.array(binned_magnitudes) + epsilon) - correction
        
        # Adjust so that minimum value is zero
        min_val = np.min(corrected_magnitudes)
        if min_val < 0:
            corrected_magnitudes -= min_val
        
        # Line plot the binned and corrected magnitude spectrum
        plt.plot(binned_frequencies, corrected_magnitudes, label=os.path.basename(wav_file_paths[idx]))

    plt.xscale('log')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    plt.ylim([0, None])
    plt.title('Adjusted FFT-based Logarithmically Binned Average Spectra for Human Hearing Range (Normalized Volumes)')
    plt.legend()
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.tight_layout()
    plt.show()
def compute_adjusted_spectrum(wav_file_path):
    """Compute the adjusted spectrum for a given audio file."""
    sample_rate, audio_data = wavfile.read(wav_file_path)
    
    # If stereo, average the channels
    if len(audio_data.shape) == 2:
        audio_data = np.mean(audio_data, axis=1)
    
    # Normalize audio
    max_amplitude = np.max(np.abs(audio_data))
    normalized_audio = (audio_data / np.max(np.abs(audio_data))) * max_amplitude
    
    # Compute the FFT-based magnitude spectrum
    frequencies, magnitude = compute_fft_spectrum(normalized_audio, sample_rate)
    
    # Filter for the typical range of human hearing (20Hz to 20kHz)
    mask = (frequencies >= 20) & (frequencies <= 20000)
    frequencies = frequencies[mask]
    magnitude = magnitude[mask]
    
    # Average the magnitudes within logarithmically spaced frequency bins
    binned_frequencies, binned_magnitudes = average_within_log_bins(frequencies, magnitude)
    
    # Correct the magnitudes based on equal loudness contours
    correction = equal_loudness_correction(binned_frequencies)
    corrected_magnitudes = 10 * np.log10(np.array(binned_magnitudes) + epsilon) - correction
    
    # Adjust so that minimum value is zero
    min_val = np.min(corrected_magnitudes)
    if min_val < 0:
        corrected_magnitudes -= min_val
    
    return binned_frequencies, corrected_magnitudes

def plot_spectrum_difference(wav_file_path_1, wav_file_path_2):
    """Visualize the difference between the spectra of two audio files."""
    freqs1, spectrum1 = compute_adjusted_spectrum(wav_file_path_1)
    _, spectrum2 = compute_adjusted_spectrum(wav_file_path_2)
    
    # Compute the difference
    difference = spectrum1 - spectrum2
    
    plt.figure(figsize=(12, 8))
    
    # Plot the spectra of both files
    plt.plot(freqs1, spectrum1, label=f'File 1: {os.path.basename(wav_file_path_1)}', linestyle='--')
    plt.plot(freqs1, spectrum2, label=f'File 2: {os.path.basename(wav_file_path_2)}', linestyle='--')
    
    # Plot the difference
    plt.plot(freqs1, difference, label='Difference', color='red', linewidth=2)
    
    plt.xscale('log')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    plt.title('Difference between Adjusted Spectra of Two Audio Files')
    plt.legend()
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

   plot_normalized_fft_average_spectra_log_binned_line_corrected_adjusted_min(['file1.wav','file2.wav'])
   plot_spectrum_difference('file.wav','file2.wav')
