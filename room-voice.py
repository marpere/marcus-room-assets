import re
import sys
import numpy as np
import scipy.io.wavfile as wav
from scipy import signal

# ===== SETTINGS =====
sample_rate = 44100
sec_per_letter = 0.11    # seconds per letter (adjust speed here)
min_word_time = 0.5      # min duration per word (short words)
pause_duration = 0.25    # pause between words
static_level = 0.05      # overall pink noise/static volume
hum_level = 0.13         # CRT hum volume
whine_level = 0.06       # CRT whine volume
drone_level = 0.12       # background drone volume (boosted for more bass)
beep_freq = 320          # BASSIER beep (Hz)
beep_gain = 0.62         # Louder beep
overall_bass_boost = 1.18 # Boost bass frequencies in final mix

def make_safe_filename(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', '_', text)
    text = re.sub(r'[^a-z0-9_]', '', text)
    return text or "room_voice"

# --- Pink noise generator (for realistic static) ---
def generate_pink_noise(length):
    n_rows = 16
    n_cols = length
    array = np.random.randn(n_rows, n_cols)
    array_cumsum = np.cumsum(array, axis=0)
    pink = array_cumsum[-1] / n_rows
    pink = pink / np.max(np.abs(pink))
    return pink

def create_continuous_bg(duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration))
    hum = hum_level * np.sin(2 * np.pi * 60 * t)
    whine = whine_level * np.sin(2 * np.pi * 800 * t)
    static = static_level * generate_pink_noise(len(t))
    drone = drone_level * (np.sin(2 * np.pi * 42 * t) +
                          0.8*np.sin(2 * np.pi * 65 * t))
    bg = hum + whine + static + drone
    bg = bg / max(1.0, np.max(np.abs(bg)))
    return bg

def bass_boost(signal, sample_rate):
    # Simple low-shelf filter to boost bass (below ~200Hz)
    from scipy.signal import lfilter, butter
    cutoff = 200 / (sample_rate / 2)
    b, a = butter(2, cutoff, btype='low')
    low = lfilter(b, a, signal)
    boosted = signal + overall_bass_boost * low
    boosted = boosted / max(1.0, np.max(np.abs(boosted)))
    return boosted

def apply_reverb_simulation(audio, sample_rate, decay_time=1.2):
    delays = [0.07, 0.18, 0.25, 0.44]
    decay_factors = [0.3, 0.22, 0.15, 0.11]
    reverb_audio = audio.copy()
    for delay, factor in zip(delays, decay_factors):
        delay_samples = int(delay * sample_rate)
        if delay_samples < len(audio):
            delayed = np.zeros_like(audio)
            delayed[delay_samples:] = audio[:-delay_samples] * factor
            reverb_audio += delayed
    return reverb_audio

def create_single_word_voice(word, sample_rate, duration, word_index, total_length, start_time):
    t_word = np.linspace(0, duration, int(sample_rate*duration))
    base_freq = 75
    voice = np.sin(2 * np.pi * base_freq * t_word)
    voice += 0.3 * np.sin(2 * np.pi * base_freq * 2 * t_word)
    voice += 0.2 * np.sin(2 * np.pi * base_freq * 3 * t_word)
    formant_cutoff = 280 / (sample_rate // 2)
    b, a = signal.butter(2, formant_cutoff, btype='low')
    voice = signal.filtfilt(b, a, voice)
    mod_freq = 4
    modulation = 1 + 0.16 * np.sin(2 * np.pi * mod_freq * t_word)
    voice = voice * modulation

    # BASSIER BEEP!
    beep = beep_gain * np.sin(2 * np.pi * beep_freq * t_word)
    beep_envelope = np.ones_like(t_word)
    attack_samples = int(0.02 * sample_rate)
    beep_envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    decay_start = duration - 0.08
    decay_start_sample = int(decay_start * sample_rate)
    beep_envelope[decay_start_sample:] = np.linspace(1, 0.04, len(beep_envelope[decay_start_sample:]))
    beep = beep * beep_envelope

    # Merge, shape, normalize
    final = (voice * 0.7 + beep)
    final = apply_reverb_simulation(final, sample_rate, decay_time=duration)
    fade_in = int(0.07 * sample_rate)
    fade_out = int(0.22 * sample_rate) if duration < 1 else int(0.37 * sample_rate)
    final[:fade_in] *= np.linspace(0, 1, fade_in)
    final[-fade_out:] *= np.linspace(1, 0, fade_out)
    final = final / np.max(np.abs(final)) * 0.8

    # Overlay word in array
    output = np.zeros(total_length)
    start_i = int(start_time * sample_rate)
    end_i = start_i + len(final)
    output[start_i:end_i] += final
    return output

# --- Main execution ---

if len(sys.argv) < 2:
    print("Usage:")
    print("  python3 room_voice.py 'your sentence here'")
    sys.exit(1)

sentence = sys.argv[1].strip()
words = sentence.split()
word_durations = [max(min_word_time, sec_per_letter * len(word)) for word in words]
total_time = sum(word_durations) + pause_duration*(len(words)-1)
output_filename = make_safe_filename(sentence) + ".wav"
total_length = int(sample_rate * total_time)

# Continuous background (static, hum, whine, drone)
background = create_continuous_bg(total_time, sample_rate)

# Bass boost the continuous background
background = bass_boost(background, sample_rate)

# Prepare word overlays
room_voice_sentence = np.zeros(total_length, dtype=np.float32)
cursor = 0.0
for i, (word, dur) in enumerate(zip(words, word_durations)):
    overlay = create_single_word_voice(word, sample_rate, dur, i, total_length, cursor)
    room_voice_sentence += overlay
    cursor += dur
    if i < len(words)-1:
        cursor += pause_duration

# Mix, normalize and write file
final_mix = (room_voice_sentence + background)
final_mix = final_mix / np.max(np.abs(final_mix)) * 0.80
final_mix_int16 = (final_mix * 32767).astype(np.int16)

wav.write(output_filename, sample_rate, final_mix_int16)
print(f"\nDone! Created: {output_filename}")
print("Each word duration (seconds):")
for word, dur in zip(words, word_durations):
    print(f"  {word}: {dur:.2f}s")
print(f"Pause between words: {pause_duration:.2f}s")
print(f"Total length: {total_time:.2f}s")
