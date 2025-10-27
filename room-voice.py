import re
import numpy as np
import scipy.io.wavfile as wav
from scipy import signal

# SETTINGS
sample_rate = 44100
sec_per_letter = 0.11   # seconds per letter (adjust for pacing)
min_word_time = 0.5     # minimum duration each word (short words)
pause_duration = 0.25   # pause between words in seconds

def make_safe_filename(text):
    # Lowercase, replace spaces with underscores, remove non-alphanumeric/underscore
    text = text.lower().strip()
    text = re.sub(r'\s+', '_', text)
    text = re.sub(r'[^a-z0-9_]', '', text)
    return text or "room_voice"

def create_crt_hum(sample_rate, duration):
    t = np.linspace(0, duration, int(sample_rate * duration))
    hum = 0.1 * np.sin(2 * np.pi * 60 * t)
    whine = 0.05 * np.sin(2 * np.pi * 800 * t)
    noise = 0.02 * np.random.normal(0, 1, len(t))
    return hum + whine + noise

def create_ominous_drone(sample_rate, duration, base_freq=42):
    t = np.linspace(0, duration, int(sample_rate * duration))
    drone = 0.3 * np.sin(2 * np.pi * base_freq * t)
    drone += 0.2 * np.sin(2 * np.pi * (base_freq * 1.5) * t)
    drone += 0.1 * np.sin(2 * np.pi * (base_freq * 2.1) * t)
    mod = 1 + 0.3 * np.sin(2 * np.pi * 0.2 * t)
    return drone * mod

def apply_reverb_simulation(audio, sample_rate, decay_time=1.5):
    delays = [0.05, 0.1, 0.2, 0.35, 0.5]
    decay_factors = [0.3, 0.25, 0.2, 0.15, 0.1]
    reverb_audio = audio.copy()
    for delay, factor in zip(delays, decay_factors):
        delay_samples = int(delay * sample_rate)
        if delay_samples < len(audio):
            delayed = np.zeros_like(audio)
            delayed[delay_samples:] = audio[:-delay_samples] * factor
            reverb_audio += delayed
    return reverb_audio

def create_single_word_room_voice(word, sample_rate, duration, word_index):
    t = np.linspace(0, duration, int(sample_rate * duration))
    base_freq = 75
    voice = np.sin(2 * np.pi * base_freq * t)
    voice += 0.3 * np.sin(2 * np.pi * base_freq * 2 * t)
    voice += 0.2 * np.sin(2 * np.pi * base_freq * 3 * t)
    formant_cutoff = 280 / (sample_rate // 2)
    b, a = signal.butter(2, formant_cutoff, btype='low')
    voice = signal.filtfilt(b, a, voice)
    mod_freq = 4
    modulation = 1 + 0.15 * np.sin(2 * np.pi * mod_freq * t)
    voice = voice * modulation

    beep_freq = 1000 + 200 * (word_index % 3)
    beep = 0.4 * np.sin(2 * np.pi * beep_freq * t)
    beep_envelope = np.ones_like(t)
    attack_samples = int(0.1 * sample_rate)
    beep_envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    decay_start = duration - 0.3
    decay_start_sample = int(decay_start * sample_rate)
    beep_envelope[decay_start_sample:] = np.linspace(1, 0.1, len(beep_envelope[decay_start_sample:]))
    beep = beep * beep_envelope

    crt_hum = create_crt_hum(sample_rate, duration)
    drone = create_ominous_drone(sample_rate, duration, base_freq=42)

    final_audio = (voice * 0.7 + beep * 0.4 + crt_hum * 0.3 + drone * 0.6)
    final_audio = apply_reverb_simulation(final_audio, sample_rate, decay_time=duration)
    fade_samples_in = int(0.1 * sample_rate)
    fade_samples_out = int(0.2 * sample_rate) if duration < 1 else int(0.4 * sample_rate)
    final_audio[:fade_samples_in] *= np.linspace(0, 1, fade_samples_in)
    final_audio[-fade_samples_out:] *= np.linspace(1, 0, fade_samples_out)
    final_audio = final_audio / np.max(np.abs(final_audio)) * 0.8
    return final_audio

# --- Script execution ---
sentence = input("Type your sentence for Room voice: ").strip()
words = sentence.split()
output_filename = f"{make_safe_filename(sentence)}.wav"

word_pause = np.zeros(int(sample_rate * pause_duration))
room_voice_sentence = np.array([], dtype=np.float32)

for i, word in enumerate(words):
    word_time = max(min_word_time, sec_per_letter * len(word))
    word_audio = create_single_word_room_voice(word, sample_rate, word_time, i)
    room_voice_sentence = np.concatenate([room_voice_sentence, word_audio])
    if i < len(words)-1:
        room_voice_sentence = np.concatenate([room_voice_sentence, word_pause])

room_voice_sentence = room_voice_sentence / np.max(np.abs(room_voice_sentence)) * 0.8
room_voice_sentence_int16 = (room_voice_sentence * 32767).astype(np.int16)

wav.write(output_filename, sample_rate, room_voice_sentence_int16)
print(f"\nDone! Created: {output_filename}")
print("Each word duration (seconds):")
for i, word in enumerate(words):
    print(f"  {word}: {max(min_word_time, sec_per_letter * len(word)):.2f}s")
print(f"Pause between words: {pause_duration:.2f}s")
