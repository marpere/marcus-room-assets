import re
import sys
import numpy as np
import scipy.io.wavfile as wav
from scipy import signal
from pydub import AudioSegment
import os

# ===== BASE SETTINGS =====
sample_rate = 44100

def mood_settings(mood):
    mood = mood.lower()
    settings = {
        "regular": {
            "sec_per_letter": 0.14,
            "min_word_time": 0.5,
            "pause_duration": 0.25,
            "beep_gain": 0.62,
            "beep_freq": 320,
            "hum_level": 0.15,
            "whine_level": 0.06,
            "drone_level": 0.14,
            "static_level": 0.025,
            "reverb_decay": 1.5,
        },
        "calm": {
            "sec_per_letter": 0.14,
            "min_word_time": 0.5,
            "pause_duration": 0.3,
            "beep_gain": 0.3,
            "beep_freq": 300,
            "hum_level": 0.1,
            "whine_level": 0.03,
            "drone_level": 0.18,
            "static_level": 0.02,
            "reverb_decay": 3.0,
        },
        "whisper": {
            "sec_per_letter": 0.17,
            "min_word_time": 0.4,
            "pause_duration": 0.3,
            "beep_gain": 0.05,
            "beep_freq": 280,
            "hum_level": 0.1,
            "whine_level": 0.01,
            "drone_level": 0.25,
            "static_level": 0.015,
            "reverb_decay": 2.8,
        },
        "angry": {
            "sec_per_letter": 0.15,
            "min_word_time": 0.4,
            "pause_duration": 0.3,
            "beep_gain": 0.75,
            "beep_freq": 380,
            "hum_level": 0.18,
            "whine_level": 0.15,
            "drone_level": 0.09,
            "static_level": 0.03,
            "reverb_decay": 1.1,
        },
        "angrier": {
            "sec_per_letter": 0.13,
            "min_word_time": 0.38,
            "pause_duration": 0.22,
            "beep_gain": 0.91,
            "beep_freq": 420,
            "hum_level": 0.21,
            "whine_level": 0.20,
            "drone_level": 0.07,
            "static_level": 0.04,
            "reverb_decay": 0.9
        },
        "furious": {
            "sec_per_letter": 0.12,
            "min_word_time": 0.35,
            "pause_duration": 0.15,
            "beep_gain": 1.10,
            "beep_freq": 480,
            "hum_level": 0.25,
            "whine_level": 0.30,
            "drone_level": 0.05,
            "static_level": 0.06,
            "reverb_decay": 0.5
        }
    }
    return settings.get(mood, settings["regular"])

def make_safe_filename(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', '_', text)
    text = re.sub(r'[^a-z0-9_]', '', text)
    return text or "room_voice"

def generate_pink_noise(length):
    n_rows = 16
    array = np.random.randn(n_rows, length)
    array_cumsum = np.cumsum(array, axis=0)
    pink = array_cumsum[-1] / n_rows
    pink = pink / np.max(np.abs(pink))
    return pink

def create_continuous_bg(duration, sample_rate, hum_level, whine_level, drone_level, static_level):
    t = np.linspace(0, duration, int(sample_rate * duration))
    hum = hum_level * np.sin(2 * np.pi * 60 * t)
    whine = whine_level * np.sin(2 * np.pi * 800 * t)
    static = static_level * generate_pink_noise(len(t))
    drone = drone_level * (np.sin(2 * np.pi * 42 * t) + 0.8*np.sin(2 * np.pi * 65 * t))
    bg = hum + whine + static + drone
    bg = bg / max(1.0, np.max(np.abs(bg)))
    return bg

def apply_reverb_simulation(audio, sample_rate, decay_time):
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

def create_single_word_voice(word, sample_rate, duration, word_index,
                             total_length, start_time, beep_gain, beep_freq, reverb_decay):
    t = np.linspace(0, duration, int(sample_rate * duration))
    base_freq = 75
    voice = np.sin(2 * np.pi * base_freq * t)
    voice += 0.3 * np.sin(2 * np.pi * base_freq * 2 * t)
    voice += 0.2 * np.sin(2 * np.pi * base_freq * 3 * t)
    formant_cutoff = 280 / (sample_rate // 2)
    b, a = signal.butter(2, formant_cutoff, btype='low')
    voice = signal.filtfilt(b, a, voice)
    mod_freq = 4
    modulation = 1 + 0.16 * np.sin(2 * np.pi * mod_freq * t)
    voice *= modulation

    beep = beep_gain * np.sin(2 * np.pi * beep_freq * t)
    beep_envelope = np.ones_like(t)
    attack_samples = int(0.02 * sample_rate)
    beep_envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    decay_start = duration - 0.08
    decay_start_sample = int(decay_start * sample_rate)
    beep_envelope[decay_start_sample:] = np.linspace(1, 0.05, len(beep_envelope[decay_start_sample:]))
    beep *= beep_envelope

    final = voice * 0.7 + beep
    final = apply_reverb_simulation(final, sample_rate, decay_time=reverb_decay)
    if beep_gain > 0.9:  # only true for angry preset above
        drive = 1.8
        clipped = np.tanh(final * drive)
        final = 0.65 * final + 0.35 * clipped
    fade_in = int(0.07 * sample_rate)
    fade_out = int(0.37 * sample_rate if duration >= 1 else 0.22 * sample_rate)
    if beep_gain > 0.9:
        fade_out = int(0.15 * sample_rate)
    else:
        fade_out = int(0.37 * sample_rate if duration >= 1 else 0.22 * sample_rate)
    final[:fade_in] *= np.linspace(0, 1, fade_in)
    final[-fade_out:] *= np.linspace(1, 0, fade_out)
    final /= np.max(np.abs(final)) + 1e-9
    final *= 0.8

    output = np.zeros(total_length)
    start_i = int(start_time * sample_rate)
    output[start_i:start_i + len(final)] += final
    return output

def convert_wav_to_mp3(wav_path, mp3_path):
    audio = AudioSegment.from_wav(wav_path)
    audio.export(mp3_path, format="mp3", bitrate="256k")

# Main execution
if len(sys.argv) < 3:
    print("Usage: python3 room_voice.py 'sentence here' mood")
    print("moods: regular, calm, whisper, angry")
    sys.exit(1)

sentence = sys.argv[1].strip()
mood = sys.argv[2].strip().lower()
settings = mood_settings(mood)

words = sentence.split()
sec_per_letter = settings["sec_per_letter"]
min_word_time = settings["min_word_time"]
pause_duration = settings["pause_duration"]
beep_gain = settings["beep_gain"]
beep_freq = settings["beep_freq"]
hum_level = settings["hum_level"]
whine_level = settings["whine_level"]
drone_level = settings["drone_level"]
static_level = settings["static_level"]
reverb_decay = settings["reverb_decay"]

word_durations = [max(min_word_time, sec_per_letter * len(word)) for word in words]
total_time = sum(word_durations) + pause_duration*(len(words) - 1)
total_length = int(sample_rate * total_time)

background = create_continuous_bg(total_time, sample_rate,
                                 hum_level, whine_level, drone_level, static_level)

room_voice_sentence = np.zeros(total_length, dtype=np.float32)
cursor = 0.0
for i, (word, dur) in enumerate(zip(words, word_durations)):
    overlay = create_single_word_voice(word, sample_rate, dur, i,
                                      total_length, cursor, beep_gain, beep_freq, reverb_decay)
    room_voice_sentence += overlay
    cursor += dur + pause_duration

final_mix = room_voice_sentence + background
final_mix /= np.max(np.abs(final_mix)) + 1e-9
final_mix *= 0.80
final_mix_int16 = (final_mix * 32767).astype(np.int16)

wav_filename = make_safe_filename(sentence) + f"_{mood}.wav"
mp3_filename = make_safe_filename(sentence) + f"_{mood}.mp3"
wav.write(wav_filename, sample_rate, final_mix_int16)

convert_wav_to_mp3(wav_filename, mp3_filename)
os.remove(wav_filename)  # Delete the WAV, keep only MP3

print(f"\nDone! Created MP3: {mp3_filename}")
print("Mood:", mood)
print("Each word duration (seconds):")
for word, dur in zip(words, word_durations):
    print(f"  {word} : {dur:.2f}")
print(f"Pause between words: {pause_duration:.2f}")
print(f"Total length: {total_time:.2f} seconds")
