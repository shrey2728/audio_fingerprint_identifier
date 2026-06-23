"""
engine.py
---------
The exact Shazam-style fingerprinting algorithm from the EE200 Q3 notebook
(EE200_Q3A_Fingerprinting.ipynb), ported verbatim into a standalone module
so the Streamlit app and the notebook agree on every function and
parameter. Nothing here changes the algorithm or its default parameters.
"""
import os
import subprocess
import pickle
import time
from collections import defaultdict

import numpy as np
from scipy.signal import get_window
from scipy.ndimage import maximum_filter


# ---------------------------------------------------------------------
# Audio I/O (ffmpeg-backed, no librosa dependency)
# ---------------------------------------------------------------------
def load_audio(path, sr=22050, mono=True, offset=0.0, duration=None):
    """Decode mp3/wav -> float32 mono signal in [-1,1] using ffmpeg as the codec backend."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: '{path}'")
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if offset > 0:
        cmd += ["-ss", str(offset)]
    cmd += ["-i", path]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-ar", str(sr), "-ac", "1" if mono else "2", "-f", "s16le", "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on '{path}': {proc.stderr.decode(errors='ignore')}")
    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    y = pcm.astype(np.float32) / 32768.0
    return y, sr


def load_audio_bytes(data: bytes, sr=22050, mono=True, offset=0.0, duration=None):
    """Same as load_audio, but decodes from in-memory bytes (for Streamlit file uploads)."""
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if offset > 0:
        cmd += ["-ss", str(offset)]
    cmd += ["-i", "pipe:0"]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-ar", str(sr), "-ac", "1" if mono else "2", "-f", "s16le", "pipe:1"]
    proc = subprocess.run(cmd, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed decoding uploaded audio: {proc.stderr.decode(errors='ignore')}")
    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    y = pcm.astype(np.float32) / 32768.0
    return y, sr


# ---------------------------------------------------------------------
# Spectrogram
# ---------------------------------------------------------------------
def stft_spectrogram(y, sr, win_length=2048, hop_length=None, window="hann"):
    """
    Plain STFT magnitude spectrogram via a sliding-window DFT (numpy/scipy only).
    Returns S (n_freq x n_frames, linear magnitude), freqs (Hz), times (s).
    """
    if hop_length is None:
        hop_length = win_length // 4

    win = get_window(window, win_length, fftbins=True)
    n_frames = 1 + (len(y) - win_length) // hop_length
    n_fft = win_length
    n_freq = n_fft // 2 + 1

    S = np.empty((n_freq, n_frames), dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_length
        frame = y[start:start + win_length] * win
        S[:, i] = np.abs(np.fft.rfft(frame, n=n_fft))

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    times = (np.arange(n_frames) * hop_length + win_length / 2) / sr
    return S, freqs, times


def to_db(S, ref=None, amin=1e-10, top_db=80.0):
    """Linear magnitude -> dB scale, clipped to top_db below the peak."""
    S = np.maximum(S, amin)
    if ref is None:
        ref = S.max()
    ref = max(ref, amin)
    S_db = 20.0 * np.log10(S / ref)
    if top_db is not None:
        S_db = np.maximum(S_db, S_db.max() - top_db)
    return S_db


# ---------------------------------------------------------------------
# Constellation map (peak picking)
# ---------------------------------------------------------------------
def find_constellation_peaks(S_db, freq_neighborhood=10, time_neighborhood=10, min_amp_db=-55.0):
    """
    Keep a point (f,t) only if it's the loudest point in a
    (2*freq_neighborhood+1) x (2*time_neighborhood+1) neighbourhood,
    and it's louder than min_amp_db (relative to the 0 dB peak of the spectrogram).
    """
    fh = 2 * freq_neighborhood + 1
    th = 2 * time_neighborhood + 1
    local_max = maximum_filter(S_db, size=(fh, th), mode="constant", cval=-np.inf)
    is_peak = (S_db == local_max) & (S_db > min_amp_db)
    freq_idx, time_idx = np.where(is_peak)
    return freq_idx, time_idx


# ---------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------
def generate_hashes(freq_idx, time_idx, fan_out=5, min_time_delta=1, max_time_delta=200,
                     freq_bits=10, time_delta_bits=10):
    """
    Pair each anchor peak with up to `fan_out` later peaks within
    [min_time_delta, max_time_delta] frames, producing (hash_key, anchor_time) pairs.
    """
    order = np.argsort(time_idx)
    f_sorted, t_sorted = freq_idx[order], time_idx[order]
    n = len(t_sorted)

    hashes = []
    for i in range(n):
        f1, t1 = f_sorted[i], t_sorted[i]
        count = 0
        for j in range(i + 1, n):
            f2, t2 = f_sorted[j], t_sorted[j]
            dt = t2 - t1
            if dt < min_time_delta:
                continue
            if dt > max_time_delta:
                break  # peaks are time-sorted, nothing further can be in range
            key = pack_hash(f1, f2, dt, freq_bits, time_delta_bits)
            hashes.append((key, int(t1)))
            count += 1
            if count >= fan_out:
                break
    return hashes


def generate_single_peak_hashes(freq_idx, time_idx):
    """Baseline: each peak's own frequency bin, with no pairing at all."""
    return [(int(f), int(t)) for f, t in zip(freq_idx, time_idx)]


def pack_hash(f1, f2, dt, freq_bits=10, time_delta_bits=10):
    """Pack (f1, f2, dt) into one integer key for O(1) dict lookups."""
    f1 = int(f1) & ((1 << freq_bits) - 1)
    f2 = int(f2) & ((1 << freq_bits) - 1)
    dt = int(dt) & ((1 << time_delta_bits) - 1)
    return (f1 << (freq_bits + time_delta_bits)) | (f2 << time_delta_bits) | dt


# ---------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------
DEFAULT_PARAMS = dict(
    sr=22050, win_length=2048, hop_length=512,
    freq_neighborhood=10, time_neighborhood=10, min_amp_db=-55.0,
    fan_out=5, min_time_delta=1, max_time_delta=200,
)


def fingerprint_audio(y, sr, params=DEFAULT_PARAMS):
    """Full pipeline: spectrogram -> constellation peaks -> paired + single-peak hashes."""
    S, freqs, times = stft_spectrogram(y, sr, win_length=params["win_length"], hop_length=params["hop_length"])
    if S.max() < 1e-9:
        # silent (or effectively silent) clip: to_db would be degenerate (every bin tied at 0 dB,
        # so every bin would register as a "peak"). There is nothing meaningful to fingerprint.
        return [], []
    S_db = to_db(S)
    freq_idx, time_idx = find_constellation_peaks(
        S_db, freq_neighborhood=params["freq_neighborhood"],
        time_neighborhood=params["time_neighborhood"], min_amp_db=params["min_amp_db"])
    paired = generate_hashes(freq_idx, time_idx, fan_out=params["fan_out"],
                              min_time_delta=params["min_time_delta"], max_time_delta=params["max_time_delta"])
    single = generate_single_peak_hashes(freq_idx, time_idx)
    return paired, single


def fingerprint_audio_with_intermediate(y, sr, params=DEFAULT_PARAMS):
    """Same as fingerprint_audio, but also returns the spectrogram/peaks for visualisation."""
    S, freqs, times = stft_spectrogram(y, sr, win_length=params["win_length"], hop_length=params["hop_length"])
    if S.max() < 1e-9:
        S_db = to_db(S)  # degenerate but still returned for display purposes
        freq_idx, time_idx = np.array([], dtype=int), np.array([], dtype=int)
        intermediate = dict(S_db=S_db, freqs=freqs, times=times, freq_idx=freq_idx, time_idx=time_idx)
        return [], [], intermediate
    S_db = to_db(S)
    freq_idx, time_idx = find_constellation_peaks(
        S_db, freq_neighborhood=params["freq_neighborhood"],
        time_neighborhood=params["time_neighborhood"], min_amp_db=params["min_amp_db"])
    paired = generate_hashes(freq_idx, time_idx, fan_out=params["fan_out"],
                              min_time_delta=params["min_time_delta"], max_time_delta=params["max_time_delta"])
    single = generate_single_peak_hashes(freq_idx, time_idx)
    intermediate = dict(S_db=S_db, freqs=freqs, times=times, freq_idx=freq_idx, time_idx=time_idx)
    return paired, single, intermediate


class FingerprintDB:
    def __init__(self, params=None):
        self.params = dict(params) if params else dict(DEFAULT_PARAMS)
        self.hash_table = defaultdict(list)
        self.single_table = defaultdict(list)
        self.songs = []

    def add_song(self, path, song_name=None, duration=None):
        if song_name is None:
            song_name = os.path.splitext(os.path.basename(path))[0]
        song_id = len(self.songs)
        self.songs.append(song_name)
        y, sr = load_audio(path, sr=self.params["sr"], duration=duration)
        paired, single = fingerprint_audio(y, sr, self.params)
        for key, t in paired:
            self.hash_table[key].append((song_id, t))
        for f, t in single:
            self.single_table[f].append((song_id, t))
        return song_id, len(paired), len(single)

    def build_from_directory(self, songs_dir, extensions=(".mp3",), duration=None, verbose=True,
                              progress_callback=None):
        files = sorted(f for f in os.listdir(songs_dir) if f.lower().endswith(extensions))
        for idx, f in enumerate(files):
            path = os.path.join(songs_dir, f)
            try:
                song_id, n_paired, n_single = self.add_song(path, duration=duration)
                if verbose:
                    print(f"  [{song_id:2d}] {self.songs[song_id]:42s} paired={n_paired:7d}  single={n_single:6d}")
            except Exception as e:
                print(f"  ERROR indexing {f}: {e}")
            if progress_callback is not None:
                progress_callback(idx + 1, len(files), f)
        return self

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump({"params": self.params, "hash_table": dict(self.hash_table),
                         "single_table": dict(self.single_table), "songs": self.songs}, fh)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        db = cls(params=data["params"])
        db.hash_table = defaultdict(list, data["hash_table"])
        db.single_table = defaultdict(list, data["single_table"])
        db.songs = data["songs"]
        return db


# ---------------------------------------------------------------------
# Matching (offset histogram)
# ---------------------------------------------------------------------
def match_paired_hashes(query_hashes, db):
    offsets_by_song = defaultdict(list)
    for key, q_t in query_hashes:
        for song_id, db_t in db.hash_table.get(key, []):
            offsets_by_song[song_id].append(db_t - q_t)
    return {sid: np.array(v) for sid, v in offsets_by_song.items()}


def match_single_peaks(query_single_hashes, db):
    offsets_by_song = defaultdict(list)
    for f, q_t in query_single_hashes:
        for song_id, db_t in db.single_table.get(f, []):
            offsets_by_song[song_id].append(db_t - q_t)
    return {sid: np.array(v) for sid, v in offsets_by_song.items()}


def score_offsets(offsets_by_song, bin_width=1):
    """Score each song by the height of the tallest bin in its offset histogram."""
    results = []
    for song_id, offs in offsets_by_song.items():
        if len(offs) == 0:
            continue
        lo, hi = offs.min(), offs.max()
        n_bins = max(1, int((hi - lo) / bin_width) + 1)
        hist, edges = np.histogram(offs, bins=n_bins, range=(lo, hi + bin_width))
        best_bin = np.argmax(hist)
        best_offset = (edges[best_bin] + edges[best_bin + 1]) / 2
        results.append(dict(song_id=song_id, score=int(hist.max()), best_offset=best_offset,
                             n_matches=len(offs), offsets=offs, histogram=(hist, edges)))
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def identify(query_paired_hashes, db, top_k=5):
    results = score_offsets(match_paired_hashes(query_paired_hashes, db))
    for r in results[:top_k]:
        r["song_name"] = db.songs[r["song_id"]]
    return results[:top_k]


def identify_single_peak(query_single_hashes, db, top_k=5):
    results = score_offsets(match_single_peaks(query_single_hashes, db))
    for r in results[:top_k]:
        r["song_name"] = db.songs[r["song_id"]]
    return results[:top_k]
