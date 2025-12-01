import numpy as np
import scipy.io.wavfile as wav
from scipy import signal
from pydub import AudioSegment
import os
import sys

# --- Configuration ---
duration = 45.0  # seconds (Length of the audio track)
sample_rate = 44100  # Hz (Standard CD Quality)
amplitude = 0.5  # Max volume for internal calculation
t = np.linspace(0., duration, int(sample_rate * duration))
BITRATE = "256k"  # High quality MP3 bitrate

# List of files to be processed
FILES_TO_PROCESS = [
    {"wav": "01_rumble_70hz.wav", "amp_scale": 0.4, "generator": "rumble"},
    {"wav": "02_ambient_drone.wav", "amp_scale": 0.3, "generator": "drone"},
    {"wav": "03_electric_hiss.wav", "amp_scale": 0.1, "generator": "hiss"}
]

# --- Helper Functions for Audio Generation ---

def generate_rumble():
    """Generates the low-frequency sine wave for physical pressure."""
    # Target frequency: 70 Hz
    return amplitude * np.sin(2. * np.pi * 70.0 * t)

def generate_drone():
    """Generates the unsettling minor interval tone layer."""
    # Target frequencies: C3 (130.81 Hz) and D#3 (155.56 Hz)
    freq1 = 130.81
    freq2 = 155.56
    drone1 = amplitude * np.sin(2. * np.pi * freq1 * t)
    drone2 = amplitude * np.sin(2. * np.pi * freq2 * t)
    return drone1 + drone2

def generate_hiss():
    """Generates the filtered white noise texture layer."""
    noise = (np.random.rand(len(t)) * 2 - 1) * amplitude
    
    # Bandpass filter for electric texture
    nyquist = 0.5 * sample_rate
    low_cutoff = 3000.0 / nyquist
    high_cutoff = 7000.0 / nyquist
    b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='band')
    hiss = signal.lfilter(b, a, noise)
    return hiss

# Map generator names to their functions
GENERATORS = {
    "rumble": generate_rumble,
    "drone": generate_drone,
    "hiss": generate_hiss
}

# --- 1. WAV Generation and Saving ---

def save_wav_file(filename, audio_data, amp_scale):
    """Normalizes and saves audio data to a 16-bit WAV file."""
    
    # Apply amplitude scaling
    audio_data *= amp_scale
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data /= max_val
    
    # Convert to 16-bit integer format
    audio_int = (audio_data * 32767).astype(np.int16)
    
    # Write the WAV file
    wav.write(filename, sample_rate, audio_int)
    print(f"✅ Successfully generated {filename}")


def create_all_wav_layers():
    """Generates all individual WAV files defined in FILES_TO_PROCESS."""
    
    print(f"--- 1. Generating WAV Layers ({duration}s, {sample_rate}Hz) ---")
    success_count = 0
    
    for file_info in FILES_TO_PROCESS:
        generator_func = GENERATORS.get(file_info["generator"])
        if generator_func:
            audio_data = generator_func()
            save_wav_file(file_info["wav"], audio_data, file_info["amp_scale"])
            success_count += 1
    
    return success_count == len(FILES_TO_PROCESS)

# --- 2. MP3 Conversion ---

def convert_wav_to_mp3(wav_path, mp3_path):
    """Converts a single WAV file to MP3 format using pydub."""
    
    if not os.path.exists(wav_path):
        return False

    try:
        audio = AudioSegment.from_wav(wav_path)
        audio.export(mp3_path, format="mp3", bitrate=BITRATE)
        print(f"✅ Converted: {wav_path} -> {mp3_path}")
        return True
    
    except FileNotFoundError:
        print("\n--- 🛑 ERROR: FFmpeg/Libav not found! ---")
        print("Conversion failed. Please install FFmpeg and ensure it's in your system PATH.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during conversion of {wav_path}: {e}")
        return False

def batch_convert_to_mp3():
    """Converts all generated WAV files to MP3."""
    
    print("\n--- 2. Starting Batch MP3 Conversion ---")
    
    for file_info in FILES_TO_PROCESS:
        wav_file = file_info["wav"]
        mp3_file = wav_file.replace(".wav", ".mp3")
        
        # Check if the WAV file exists before trying to convert
        if os.path.exists(wav_file):
            convert_wav_to_mp3(wav_file, mp3_file)
        else:
            print(f"⚠️ Skipping conversion for {wav_file} (File not found).")
    
    print("\nProcess complete.")

# --- Main Execution ---
if __name__ == "__main__":
    
    # Step 1: Generate the raw audio files
    if create_all_wav_layers():
        # Step 2: Convert the generated files
        batch_convert_to_mp3()
    else:
        print("\nFATAL ERROR: Could not generate all WAV layers. Stopping script.")