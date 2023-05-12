'''
Guitar tuner script based on the Harmonic Product Spectrum (HPS)

MIT License
Copyright (c) 2021 chciken

https://www.chciken.com/digital/signal/processing/2020/05/13/guitar-tuner.html#dft

https://newt.phys.unsw.edu.au/jw/notes.html

'''

import copy
import os
import numpy as np
import scipy.fftpack
import sounddevice as sd
import time

# General settings that can be changed by the user
SAMPLE_FREQ = 48000 # sample frequency in Hz
WINDOW_SIZE = 48000 # window size of the DFT in samples
WINDOW_STEP = 12000 # step size of window
NUM_HPS = 5 # max number of harmonic product spectrums
POWER_THRESH = 1e-6 # tuning is activated if the signal power exceeds this threshold
CONCERT_PITCH = 440 # defining a1
WHITE_NOISE_THRESH = 0.2 # everything under WHITE_NOISE_THRESH*avg_energy_per_freq is cut off

WINDOW_T_LEN = WINDOW_SIZE / SAMPLE_FREQ # length of the window in seconds
SAMPLE_T_LENGTH = 1 / SAMPLE_FREQ # length between two samples in seconds
DELTA_FREQ = SAMPLE_FREQ / WINDOW_SIZE # frequency step width of the interpolated DFT
OCTAVE_BANDS = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]


ALL_NOTES = ["A","A+","A#","B-","B","B+","C","C+","C#","D-","D","D+","D#","E-","E","E+","F","F+","F#","G-","G","G+","G#","A-"]
NOTE_OUD_POSITION_NAME = {
  "C2" :"Karar Rast", "C+2" :"Karar Nim Zirkola", "C#2" :"Karar Zirkola", "D-2" :"Karar Nim Douka", "D2" :"Karar Douka", "D+2" :"Karar Nim Kurd", "D#2" :"Karar Kurd", "E-2" :"Karar Sika", "E2" :"Karar Bousalik", "E+2":"Karar Tik Bousalik", "F2":"Karar Jaharika", "F+2": "Karar Nim Hijaz", "F#2" :"Karar Hijaz", "G-1":"Nim Yaka",
  "G2" :"Yaka", "G+2" :"Karar Nim Hisar", "G#2" :"Karar Hisar", "A-2" :"Nim Oshayran",
  "A2" :"Oshayran", "A+2" :"Nim Ajam Oshayran", "A#2" :"Ajam Oshayran", "B-2" :"Irak", "B2" :"Kawasht", "B+2" :"Tik Kawasht", "C3" :"Rast", "C+3" :"Nim Zirkola", "C#3" :"Zirkola", "D-3" :"Nim Douka", 
  "D3" :"Douka", "D+3" :"Nim Kurd", "D#3" :"Kurd", "E-3" :"Sika", "E3" :"Bousalik", "E+3" :"Tik Bousalik", "F3" :"Jaharika", "F+3" :"Nim Hijaz", "F#3" :"Hejaz (a.k.a Saba)", "G-3" :"Nim Nawa",
  "G3" :"Nawa", "G+3" :"Nim Hisar", "G#3" :"Hisar", "A-3" :"Nim Housayni", "A3" :"Housayni", "A+3" :"Tik Housayni", "A#3" :"Ajam", "B-3" :"Awj", "B3" :"Mahour", "B+3" :"Nim Kerdan", 
  "C4" :"Kerdan", "C+4" :"Nim Shahnaz", "C#4" :"Shahnaz", "D-4" :"Nim Mouhayyar", "D4" :"Mouhayyar", "D+4" :"Tik Mouhayyar", "D#4" :"Sonboula", "E-4" :"Bazrak", "E4" :"Jawab Bousalik", "E+4":"Nim Mahouran", "F4": "Mahouran","F+4":"Tik Mahouran", "F#4" : "Jawab Hijaz", "G-4":"Nim Sahm", "G4":"Sahm"
}
def find_closest_note(pitch):

  i = int(np.round(np.log2(pitch/CONCERT_PITCH)*24))
  closest_note = ALL_NOTES[i%24] + str(4 + (i + 18) // 24)
  closest_pitch = CONCERT_PITCH*2**(i/24)
  return closest_note, closest_pitch, i

HANN_WINDOW = np.hanning(WINDOW_SIZE)
def callback(indata, frames, time, status):
  """
  Callback function of the InputStream method.
  That's where the magic happens ;)
  """
  # define static variables
  if not hasattr(callback, "window_samples"):
    callback.window_samples = [0 for _ in range(WINDOW_SIZE)]
  if not hasattr(callback, "noteBuffer"):
    callback.noteBuffer = ["1","2"]

  if status:
    print(status)
    return
  if any(indata):
    callback.window_samples = np.concatenate((callback.window_samples, indata[:, 0])) # append new samples
    callback.window_samples = callback.window_samples[len(indata[:, 0]):] # remove old samples

    # skip if signal power is too low
    signal_power = (np.linalg.norm(callback.window_samples, ord=2)**2) / len(callback.window_samples)
    if signal_power < POWER_THRESH:
      os.system('cls' if os.name=='nt' else 'clear')
      print("Closest note: ...")
      return

    # avoid spectral leakage by multiplying the signal with a hann window
    hann_samples = callback.window_samples * HANN_WINDOW
    magnitude_spec = abs(scipy.fftpack.fft(hann_samples)[:len(hann_samples)//2])

    # supress mains hum, set everything below 62Hz to zero
    for i in range(int(62/DELTA_FREQ)):
      magnitude_spec[i] = 0

    # calculate average energy per frequency for the octave bands
    # and suppress everything below it
    for j in range(len(OCTAVE_BANDS)-1):
      ind_start = int(OCTAVE_BANDS[j]/DELTA_FREQ)
      ind_end = int(OCTAVE_BANDS[j+1]/DELTA_FREQ)
      ind_end = ind_end if len(magnitude_spec) > ind_end else len(magnitude_spec)
      avg_energy_per_freq = (np.linalg.norm(magnitude_spec[ind_start:ind_end], ord=2)**2) / (ind_end-ind_start)
      avg_energy_per_freq = avg_energy_per_freq**0.5
      for i in range(ind_start, ind_end):
        magnitude_spec[i] = magnitude_spec[i] if magnitude_spec[i] > WHITE_NOISE_THRESH*avg_energy_per_freq else 0

    # interpolate spectrum
    mag_spec_ipol = np.interp(np.arange(0, len(magnitude_spec), 1/NUM_HPS), np.arange(0, len(magnitude_spec)),
                              magnitude_spec)
    mag_spec_ipol = mag_spec_ipol / np.linalg.norm(mag_spec_ipol, ord=2) #normalize it

    hps_spec = copy.deepcopy(mag_spec_ipol)

    # calculate the HPS
    for i in range(NUM_HPS):
      tmp_hps_spec = np.multiply(hps_spec[:int(np.ceil(len(mag_spec_ipol)/(i+1)))], mag_spec_ipol[::(i+1)])
      if not any(tmp_hps_spec):
        break
      hps_spec = tmp_hps_spec

    max_ind = np.argmax(hps_spec)
    max_freq = max_ind * (SAMPLE_FREQ/WINDOW_SIZE) / NUM_HPS

    closest_note, closest_pitch, i = find_closest_note(max_freq)
    max_freq = round(max_freq, 1)
    closest_pitch = round(closest_pitch, 1)

    callback.noteBuffer.insert(0, closest_note) # note that this is a ringbuffer
    callback.noteBuffer.pop()

    os.system('cls' if os.name=='nt' else 'clear')
    if callback.noteBuffer.count(callback.noteBuffer[0]) == len(callback.noteBuffer):
      print(f"Closest note: {closest_note} {max_freq}/{closest_pitch}")
      print("Note poistion name: ", end="")
      if closest_note in NOTE_OUD_POSITION_NAME:
        print(NOTE_OUD_POSITION_NAME[closest_note])
      else:
        print("?")
    else:
      print(f"Closest note: ...")

  else:
    print('no input')

try:
  print("Starting HPS guitar tuner...")
  with sd.InputStream(channels=1, callback=callback, blocksize=WINDOW_STEP, samplerate=SAMPLE_FREQ):
    while True:
      time.sleep(0.5)
except Exception as exc:
  print(str(exc))