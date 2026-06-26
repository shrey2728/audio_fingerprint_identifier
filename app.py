"""
EE200 Q3-B — Streamlit app for the Shazam-style audio fingerprinting identifier.

Two modes:
  1. Single-clip mode: upload one query clip, see the spectrogram, the
     constellation map, the offset histogram, and the identified song.
  2. Batch mode: upload many query clips, get back a results.csv with
     exactly two columns: filename, prediction.

The fingerprinting algorithm itself (engine.py) is an exact, unmodified
port of the algorithm in EE200_Q3A_Fingerprinting.ipynb.
"""
import os
import io
import time
import zipfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from engine import (
    DEFAULT_PARAMS,
    FingerprintDB,
    load_audio,
    load_audio_bytes,
    fingerprint_audio,
    fingerprint_audio_with_intermediate,
    identify,
    identify_single_peak,
)

# -----------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Audio Fingerprint Identifier",
    page_icon="🎵",
    layout="wide",
)

SONGS_DIR = "songs"
DB_PATH = "fingerprint_db.pkl"


@st.cache_resource(show_spinner=False)
def get_database():
    """Load the prebuilt database shipped with the app, or build it on first run."""
    if os.path.exists(DB_PATH):
        return FingerprintDB.load(DB_PATH)
    if not os.path.isdir(SONGS_DIR):
        return None
    db = FingerprintDB()
    progress_bar = st.progress(0.0, text="Indexing song database...")

    def _cb(done, total, name):
        progress_bar.progress(done / total, text=f"Indexing {done}/{total}: {name}")

    db.build_from_directory(SONGS_DIR, progress_callback=_cb)
    db.save(DB_PATH)
    progress_bar.empty()
    return db


def plot_spectrogram(S_db, freqs, times, title, max_freq=5000):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.imshow(
        S_db, aspect="auto", origin="lower",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        cmap="magma", vmin=-80, vmax=0,
    )
    ax.set_ylim(0, max_freq)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_constellation(S_db, freqs, times, freq_idx, time_idx, title, max_freq=5000):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.imshow(
        S_db, aspect="auto", origin="lower",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        cmap="gray_r", vmin=-80, vmax=0, alpha=0.5,
    )
    ax.scatter(times[time_idx], freqs[freq_idx], s=10, c="red", marker="o")
    ax.set_ylim(0, max_freq)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"{title} ({len(freq_idx)} peaks)")
    fig.tight_layout()
    return fig


def plot_offset_histogram(results, top_n=3):
    n = min(top_n, len(results))
    if n == 0:
        return None
    fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, results[:n]):
        offs = r["offsets"]
        best = r["best_offset"]
        window = max(50, int(0.02 * (offs.max() - offs.min() + 1)))
        lo, hi = best - window, best + window
        hist, edges = np.histogram(offs, bins=min(200, 2 * window + 1), range=(lo, hi))
        ax.bar(edges[:-1], hist, width=(edges[1] - edges[0]), color="steelblue")
        ax.set_title(f"{r['song_name']}\nscore={r['score']} (n_matches={r['n_matches']})", fontsize=10)
        ax.set_xlabel(f"Offset (frames), ±{window}")
        ax.set_ylabel("Count")
    fig.suptitle("Offset histogram — a sharp single spike means a true match", y=1.05)
    fig.tight_layout()
    return fig


def plot_score_comparison(results_paired, results_single):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    names_p = [r["song_name"] for r in results_paired][:5]
    scores_p = [r["score"] for r in results_paired][:5]
    axes[0].barh(names_p[::-1], scores_p[::-1], color="seagreen")
    axes[0].set_xlabel("Score (tallest offset-histogram bin)")
    axes[0].set_title("Paired hashes (f1, f2, Δt)")

    names_s = [r["song_name"] for r in results_single][:5]
    scores_s = [r["score"] for r in results_single][:5]
    axes[1].barh(names_s[::-1], scores_s[::-1], color="indianred")
    axes[1].set_xlabel("Score (tallest offset-histogram bin)")
    axes[1].set_title("Single (unpaired) peaks")

    fig.suptitle("Top-5 matches: paired hashes vs. single peaks", y=1.05)
    fig.tight_layout()
    return fig


def run_identification(y, sr, db, params=DEFAULT_PARAMS):
    """Run the full pipeline on one clip and return everything needed for display."""
    min_samples = params["win_length"]
    if len(y) < min_samples:
        raise ValueError(
            f"Clip is too short to analyse (need at least "
            f"{min_samples/sr:.2f}s of audio, got {len(y)/sr:.2f}s)."
        )
    paired, single, intermediate = fingerprint_audio_with_intermediate(y, sr, params)
    results_paired = identify(paired, db, top_k=5)
    results_single = identify_single_peak(single, db, top_k=5)
    return dict(
        paired=paired, single=single, intermediate=intermediate,
        results_paired=results_paired, results_single=results_single,
    )


# -----------------------------------------------------------------------
# Sidebar / shared state
# -----------------------------------------------------------------------
st.title("🎵 Audio Fingerprint Identifier")
st.caption(
    "A from-scratch Shazam-style song identifier: spectrogram → constellation map → "
    "paired-hash fingerprints → offset-histogram matching. (EE200 Q3-B)"
)

with st.spinner("Loading song database..."):
    db = get_database()

if db is None:
    st.error(
        f"No song database found. Make sure a `{DB_PATH}` file or a `{SONGS_DIR}/` "
        "folder of .mp3 files is shipped alongside this app."
    )
    st.stop()

st.sidebar.header("Database")
st.sidebar.success(f"{len(db.songs)} songs indexed")
with st.sidebar.expander("Show indexed songs"):
    for name in db.songs:
        st.write(f"- {name}")

mode = st.sidebar.radio("Mode", ["Single-clip mode", "Batch mode"])

st.sidebar.divider()
st.sidebar.caption(
    "Fingerprinting parameters (fixed, matching the report):\n\n"
    f"- Sample rate: {DEFAULT_PARAMS['sr']} Hz\n"
    f"- Window: {DEFAULT_PARAMS['win_length']} samples\n"
    f"- Hop: {DEFAULT_PARAMS['hop_length']} samples\n"
    f"- Peak neighbourhood: ±{DEFAULT_PARAMS['freq_neighborhood']} bins / "
    f"±{DEFAULT_PARAMS['time_neighborhood']} frames\n"
    f"- Min peak loudness: {DEFAULT_PARAMS['min_amp_db']} dB\n"
    f"- Fan-out: {DEFAULT_PARAMS['fan_out']} pairs/anchor"
)
import numpy as np
import streamlit as st
from scipy.interpolate import interp1d

# =========================================================================
# 1. ADD ROBUSTNESS HELPER FUNCTIONS
# =========================================================================

def add_white_noise(y, noise_level_db):
    """
    Adds additive white Gaussian noise to the query waveform based on target dB level.
    If noise_level_db <= -80, the audio is considered clean and left unaltered.
    """
    if noise_level_db <= -80:
        return y
    
    # Calculate power of the input audio signal
    sig_power = np.mean(y ** 2)
    if sig_power == 0:
        sig_power = 1e-6
        
    # Translate target dB level to relative noise power variance
    noise_power = 10 ** (noise_level_db / 10.0)
    noise = np.random.normal(0, np.sqrt(noise_power), len(y))
    return y + noise

def apply_pitch_shift(y, sr, semitones):
    """
    Applies a simple time-domain resampling pitch shift.
    Resampling changes pitch and duration concurrently, accurately modeling 
    analog speed/pitch variations common in real-world environments.
    """
    if semitones == 0:
        return y
    
    # Calculate resampling conversion factor
    factor = 2.0 ** (semitones / 12.0)
    
    x_old = np.arange(len(y))
    x_new = np.arange(0, len(y), factor)
    
    # Interpolate across the resampled timeline
    f_interp = interp1d(x_old, y, kind='linear', fill_value="extrapolate")
    return f_interp(x_new)


# =========================================================================
# 2. RENDER THE INTERACTIVE SIDEBAR PANELS
# =========================================================================

st.sidebar.header("Robustness Testing Panel")
st.sidebar.markdown("Manipulate channel conditions to see how the algorithm handles impairments:")

# Interactive White Noise injection slider
noise_level = st.sidebar.slider(
    "Additive White Noise (dB)", 
    min_value=-80, 
    max_value=0, 
    value=-80, 
    step=5,
    help="Set to -80 for standard clean audio. Higher values add more heavy noise."
)

# Interactive Pitch Shift slider 
pitch_semitones = st.sidebar.slider(
    "Pitch Shift (Semitones)", 
    min_value=-3, 
    max_value=3, 
    value=0, 
    step=1,
    help="Positive values raise pitch and slightly speed up playback. Negative values lower them."
)


# =========================================================================
# 3. INTERCEPT AND PREPROCESS THE AUDIO PIPELINE
# =========================================================================

# Locate the segment where you fetch your query waveform array, for example:
# y, sr = load_audio(uploaded_file, offset=query_offset, duration=query_duration)

if 'y' in locals() and y is not None:
    # 1. Apply any selected Pitch Shift modifications first
    if pitch_semitones != 0:
        y = apply_pitch_shift(y, sr, pitch_semitones)
        st.write(f"🔄 **Applied Pitch Shift Mod:** {pitch_semitones} semitone(s)")

    # 2. Mix in Additive White Gaussian Noise
    if noise_level > -80:
        y = add_white_noise(y, noise_level)
        st.write(f"🔊 **Injected Noise Level:** {noise_level} dB")
        
    # --- Now proceed with your standard fingerprint matching below ---
    # S_db = compute_spectrogram(y, ...)
    # freq_idx, time_idx = find_constellation_peaks(S_db, ...)
    # ...
# -----------------------------------------------------------------------
# Single-clip mode
# -----------------------------------------------------------------------
if mode == "Single-clip mode":
    st.header("Single-clip identification")
    st.write("Upload one short audio clip (a few seconds is enough) to identify it.")

    uploaded = st.file_uploader(
        "Upload a query clip", type=["mp3", "wav", "m4a", "ogg", "flac"], key="single_upload"
    )

    if uploaded is not None:
        audio_bytes = uploaded.getvalue()
        st.audio(audio_bytes)

        with st.spinner("Decoding and fingerprinting the clip..."):
            try:
                y, sr = load_audio_bytes(audio_bytes, sr=DEFAULT_PARAMS["sr"])
            except Exception as e:
                st.error(f"Could not decode this file: {e}")
                st.stop()

            try:
                t0 = time.time()
                out = run_identification(y, sr, db)
                elapsed = time.time() - t0
            except ValueError as e:
                st.error(str(e))
                st.stop()

        results_paired = out["results_paired"]
        results_single = out["results_single"]
        intermediate = out["intermediate"]

        if not results_paired:
            st.warning("No matches found at all — try a longer or clearer clip.")
        else:
            best = results_paired[0]
            runner_up_score = results_paired[1]["score"] if len(results_paired) > 1 else 0
            confident = best["score"] >= 10 and best["score"] >= 3 * max(runner_up_score, 1)

            st.subheader("🎯 Identified song")
            if confident:
                st.success(f"**{best['song_name']}**  (score={best['score']}, computed in {elapsed*1000:.0f} ms)")
            else:
                st.warning(
                    f"Best guess: **{best['song_name']}**  (score={best['score']}) — "
                    "but the match is weak/ambiguous; try a longer or cleaner clip."
                )

            st.subheader("Top-5 candidates")
            df = pd.DataFrame(
                [{"Rank": i + 1, "Song": r["song_name"], "Score": r["score"], "Hash matches": r["n_matches"]}
                 for i, r in enumerate(results_paired)]
            )
            st.dataframe(df, hide_index=True, use_container_width=True)

            st.subheader("Intermediate steps")
            tabs = st.tabs(["Spectrogram", "Constellation map", "Offset histogram", "Single peaks vs. paired hashes"])

            with tabs[0]:
                fig = plot_spectrogram(
                    intermediate["S_db"], intermediate["freqs"], intermediate["times"],
                    f"Spectrogram of the uploaded clip",
                )
                st.pyplot(fig)

            with tabs[1]:
                fig = plot_constellation(
                    intermediate["S_db"], intermediate["freqs"], intermediate["times"],
                    intermediate["freq_idx"], intermediate["time_idx"],
                    "Constellation map of the uploaded clip",
                )
                st.pyplot(fig)

            with tabs[2]:
                fig = plot_offset_histogram(results_paired, top_n=3)
                if fig:
                    st.pyplot(fig)
                st.caption(
                    "A correct match produces a single tall, narrow spike (all matching "
                    "hashes agree on one time offset). A wrong song produces scattered, "
                    "near-uniform matches across many offsets."
                )

            with tabs[3]:
                fig = plot_score_comparison(results_paired, results_single)
                st.pyplot(fig)
                p_top = results_paired[0]["score"]
                p_second = results_paired[1]["score"] if len(results_paired) > 1 else 1
                s_top = results_single[0]["score"]
                s_second = results_single[1]["score"] if len(results_single) > 1 else 1
                st.caption(
                    f"Paired-hash margin: {p_top}/{max(p_second,1)} = {p_top/max(p_second,1):.1f}x  •  "
                    f"Single-peak margin: {s_top}/{max(s_second,1)} = {s_top/max(s_second,1):.1f}x  "
                    "— pairing makes the decision far more decisive."
                )

# -----------------------------------------------------------------------
# Batch mode
# -----------------------------------------------------------------------
else:
    st.header("Batch identification")
    st.write(
        "Upload multiple query clips (or a single .zip containing them) to identify "
        "all of them at once and download a `results.csv` with exactly two columns: "
        "`filename, prediction`."
    )

    uploaded_files = st.file_uploader(
        "Upload query clips",
        type=["mp3", "wav", "m4a", "ogg", "flac", "zip"],
        accept_multiple_files=True,
        key="batch_upload",
    )

    if uploaded_files:
        # expand any .zip uploads into individual (filename, bytes) pairs
        clips = []
        for f in uploaded_files:
            if f.name.lower().endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(f.getvalue())) as zf:
                    for name in zf.namelist():
                        if name.lower().endswith((".mp3", ".wav", ".m4a", ".ogg", ".flac")) and not name.endswith("/"):
                            clips.append((os.path.basename(name), zf.read(name)))
            else:
                clips.append((f.name, f.getvalue()))

        st.write(f"**{len(clips)} clip(s)** ready to process.")

        if st.button("Run batch identification", type="primary"):
            progress_bar = st.progress(0.0)
            status = st.empty()
            rows = []
            for i, (name, data) in enumerate(clips):
                status.write(f"Processing `{name}` ({i+1}/{len(clips)})...")
                try:
                    y, sr = load_audio_bytes(data, sr=DEFAULT_PARAMS["sr"])
                    paired, _single = fingerprint_audio(y, sr)
                    results = identify(paired, db, top_k=1)
                    prediction = results[0]["song_name"] if results else ""
                except Exception as e:
                    prediction = ""
                    st.warning(f"Failed on `{name}`: {e}")
                rows.append({"filename": name, "prediction": prediction})
                progress_bar.progress((i + 1) / len(clips))

            status.empty()
            progress_bar.empty()

            results_df = pd.DataFrame(rows, columns=["filename", "prediction"])
            st.subheader("Results")
            st.dataframe(results_df, hide_index=True, use_container_width=True)

            csv_bytes = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download results.csv",
                data=csv_bytes,
                file_name="results.csv",
                mime="text/csv",
                type="primary",
            )
