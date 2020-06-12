# python3

import collections
import dataclasses
import os
import pickle
import time

from absl import app
from absl import flags

import numpy as np
import pyaudio
import sklearn.ensemble

flags.DEFINE_enum('mode', 'collect', ['collect', 'fit', 'test', 'exercise'], 'What to execute')
flags.DEFINE_float('duration', float('inf'), 'Listening duration.')
flags.DEFINE_integer('sample_rate', 44100, 'Sampling rate.')
flags.DEFINE_float('min_freq', 70., 'Min frequency.')
flags.DEFINE_float('max_freq', 400., 'Max frequency.')
flags.DEFINE_float('window', 0.1, 'Sample window (in seconds)')
flags.DEFINE_float('db_thresh', 55, 'Decibel threshold')
flags.DEFINE_string('root_dir', '.', 'Root directory.')
flags.DEFINE_integer('num_chords', 0, 'Number of chords to recognize.')


@dataclasses.dataclass
class Config:
  duration: float
  sample_rate: int
  window: float
  min_freq: float
  max_freq: float
  db_thresh: float
  root_dir: str
  num_chords: int


def listen(config):
  audio = pyaudio.PyAudio()
  stream = audio.open(
      rate=config.sample_rate, channels=1,
      format=pyaudio.paInt16,
      input=True)
  stream.start_stream()
  read_frames = int(config.window * config.sample_rate)
  start = time.time()
  while time.time() - start < config.duration:
    in_data = stream.read(read_frames)
    yield np.frombuffer(in_data, dtype=np.int16)


def fit(config):
  assert config.num_chords, 'Must specify the number of chords.'
  data = []
  with open(os.path.join(config.root_dir, 'collection.npz'), 'rb') as f:
    size = os.fstat(f.fileno()).st_size
    while f.tell() < size:
      data.append(np.load(f))
  data = np.asarray(data)
  xs = data[:, :-(config.num_chords + 1)]
  ys = np.argmax(data[:, -(config.num_chords + 1):], axis=-1)
  model = sklearn.ensemble.GradientBoostingClassifier()
  model.fit(xs, ys)
  with open(os.path.join(config.root_dir, 'model.pkl'), 'wb') as f:
    pickle.dump(model, f)


def features(config, frames):
  freqs = np.fft.rfftfreq(len(frames), d=1 / config.sample_rate)
  db = 10 * np.log10(abs(np.fft.rfft(frames)) + 1e-8)
  db *= db > config.db_thresh
  return db[(freqs >= config.min_freq) | (freqs <= config.max_freq)]


def countdown():
  for t in range(5, 0, -1):
    print(t, end=' ', flush=True)
    time.sleep(1.)
  print('Go!', flush=True)


def collect(config):
  assert np.isfinite(config.duration)
  assert config.num_chords
  with open(os.path.join(config.root_dir, 'collection.npz'), 'wb') as f:
    for i in range(config.num_chords + 1):
      label = np.zeros(config.num_chords + 1)
      label[i] = 1.
      if i == 0:
        print('Listening to background. Don\'t play.', flush=True)
      else:
        print('Play chord #{}'.format(i), flush=True)
      countdown()
      for window in listen(config):
        feat = features(config, window)
        if feat is None:
          continue
        np.save(f, np.r_[feat, label])


def test(config):
  config.duration = float('inf')
  with open(os.path.join(config.root_dir, 'model.pkl'), 'rb') as f:
    model = pickle.load(f)
  for window in listen(config):
    feat = features(config, window)
    if feat is None:
      continue
    print(model.predict(feat.reshape(1, -1))[0])


def exercise(config):
  with open(os.path.join(config.root_dir, 'model.pkl'), 'rb') as f:
    model = pickle.load(f)
  switches = [None]
  cur = None
  cur_count = 0
  countdown()
  for window in listen(config):
    feat = features(config, window)
    if feat is None:
      continue
    prediction = model.predict(feat.reshape(1, -1))[0]
    if prediction == 0:  # Background
      continue
    if cur == prediction:
      cur_count += 1
      if cur_count > 5 and switches[-1] != cur:
        switches.append(cur)
        if len(switches) > 1:
          print('{} -> {}'.format(*switches[-2:]))
    else:
      cur = prediction
      cur_count = 1
  print('Total switches: {}'.format(len(switches) - 1))


def main(argv):
  del argv  # Unused
  config = Config(
      duration=flags.FLAGS.duration,
      sample_rate=flags.FLAGS.sample_rate,
      window=flags.FLAGS.window,
      min_freq=flags.FLAGS.min_freq,
      max_freq=flags.FLAGS.max_freq,
      db_thresh=flags.FLAGS.db_thresh,
      root_dir=flags.FLAGS.root_dir,
      num_chords=flags.FLAGS.num_chords)

  dict(
      collect=collect,
      fit=fit,
      test=test,
      exercise=exercise
  )[flags.FLAGS.mode](config)


if __name__ == '__main__':
  app.run(main)
