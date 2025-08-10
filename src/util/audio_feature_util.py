import librosa, os, pathlib, joblib
from sklearn.preprocessing import MinMaxScaler
from util import file_util

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def retrieve_audio_features(x, sr):
  return {
    "zero_crossing": librosa.feature.zero_crossing_rate(y=x),
    "root_mean_square": librosa.feature.rms(y=x),
    "spectral_centroid": librosa.feature.spectral_centroid(y=x, sr=sr),
    "spectral_bandwidth": librosa.feature.spectral_bandwidth(y=x, sr=sr),
    "spectral_rolloff": librosa.feature.spectral_rolloff(y=x, sr=sr),
    "spectral_flatness": librosa.feature.spectral_flatness(y=x),
    "spectral_contrast": librosa.feature.spectral_contrast(y=x, sr=sr),
    "rolloff": librosa.feature.spectral_rolloff(y=x, sr=sr),
    "tempo": librosa.feature.tempo(y=x, sr=sr),
    "tempogram_ratio": librosa.feature.tempogram_ratio(y=x, sr=sr),
    "chroma_stft": librosa.feature.chroma_stft(y=x, sr=sr),
    "mfcc": librosa.feature.mfcc(y=x, sr=sr)
  }
  
def create_spectrogram(data, sr, display=True, save: pathlib.Path=None):
  melspectrogram = librosa.feature.melspectrogram(y=data, sr=sr)
  melspectrogram_db = librosa.power_to_db(melspectrogram, ref=np.max)
  
  with plt.ioff():
    fig, ax = plt.subplots(ncols=1, nrows=1)
    librosa.display.specshow(melspectrogram_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    
    for spine in ax.spines.values():
      spine.set_visible(False)
    ax.set_axis_off()
    
    if save is not None:
      plt.savefig(save, bbox_inches='tight', pad_inches=0)
      
    if display:
      plt.show()
      
    plt.close("all")
  
  
def _load_audio_features(path: pathlib.Path, max_duration, discard_short_audio, with_melspectrograms, melspectrograms_save_path: pathlib.Path):
  x, sr = librosa.load(path, duration=max_duration)
  if discard_short_audio and librosa.get_duration(y=x, sr=sr) < max_duration:
    return (None, None)
  
  audio_features = retrieve_audio_features(x, sr)
  if with_melspectrograms:
    file_util.create_path(melspectrograms_save_path)
    create_spectrogram(x, sr, display=False, save=melspectrograms_save_path / pathlib.Path(path.stem).stem.replace(".", ""))
  
  return (path.stem, audio_features)
  
  
def load_audio_features(path: pathlib.Path, max_duration=30, thread_pool_size=10, discard_short_audio=False, with_melspectrograms=True, melspectrograms_save_path: pathlib.Path=None):
  audio_features = {}
  
  path_files = []
  for root, dirs, files in os.walk(path):
    for file in files:
      if file.endswith((".mp3", ".ma4", ".wav", ".ogg")):
        path_files.append(file)
    break
  
  for file_index in range(0, len(path_files), thread_pool_size):
    jobs = [
      joblib.delayed(_load_audio_features)(path / path_files[i], max_duration, discard_short_audio, with_melspectrograms, melspectrograms_save_path) 
      for i in range(file_index, min(len(path_files), file_index+thread_pool_size))
    ]
    results = joblib.Parallel(len(jobs), verbose=1)(jobs)
    
    for key, value in results:
      if discard_short_audio and ((key is None) or (value is None)):
        continue
      audio_features[key] = value
  
  return audio_features

def format_features(audio_features: dict):
  # Format features for use in dataframe.
  formated_features = []
  for filename, features in audio_features.items():
    feature_item = {"filename":filename}
    
    for key, feature in features.items():
      if (feature.shape[0] > 1):# if dimension of ndarray > 1, split feature into a subfeature
        for subfeature_index in range(feature.shape[0]):
          subfeature_key = key + "_" + str(subfeature_index)
          feature_item[subfeature_key + "_mean"] = feature[subfeature_index].mean()
          feature_item[subfeature_key + "_var"] = feature[subfeature_index].var()
      else:
        feature_item[key + "_mean"] = feature.mean()
        feature_item[key + "_var"] = feature.var()
        
    formated_features.append(feature_item)
  return formated_features

def normalize(data, range=(-1, 1), relative_to=None):
  scaler = MinMaxScaler(feature_range=range)
  if relative_to is not None:
    scaler.fit(relative_to)
    return scaler.transform(data)
  else:
    return scaler.fit_transform(data)
  
  
def normalize_audio_features(df, relative_to=None):
  filenames = df["filename"]
  only_data_df = df.drop("filename", axis=1)
  
  normalized_df = pd.DataFrame(data=normalize(only_data_df, relative_to=relative_to), columns=only_data_df.columns)
  normalized_df.insert(loc=0, column="filename", value=filenames)
  return normalized_df

  
def audio_features_as_dataframe(features):
  formated_features = format_features(features)
  df = pd.DataFrame(data=formated_features)
  
  filenames = df["filename"]
  only_data_df = df.drop("filename", axis=1)
  
  rearanged_df = pd.DataFrame(data=only_data_df, columns=only_data_df.columns)
  rearanged_df.insert(loc=0, column="filename", value=filenames)
  return rearanged_df
        
def save_audio_features_as_csv(features, save_path: pathlib.Path):
  audio_features_as_dataframe(features).to_csv(path_or_buf=save_path, index=False)