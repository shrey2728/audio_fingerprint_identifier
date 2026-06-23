# EE200 Q3-B — Audio Fingerprint Identifier (Streamlit app)

A Shazam-style song identifier wrapped in a small interactive app. The
fingerprinting algorithm (`engine.py`) is an exact, unmodified port of the
algorithm developed in `EE200_Q3A_Fingerprinting.ipynb`: STFT spectrogram →
local-maxima constellation map → paired `(f1, f2, Δt)` hashes → offset-
histogram matching.

## Files

```
app.py                 Streamlit app (UI, two modes)
engine.py              Fingerprinting algorithm (ported from the Part-A notebook)
fingerprint_db.pkl     Prebuilt hash database for the 50 provided songs
songs/                 (optional) the original mp3s — NOT required at runtime,
                       only used to rebuild fingerprint_db.pkl from scratch
requirements.txt       Python dependencies
packages.txt           System dependency (ffmpeg) for Streamlit Cloud
```

## Running locally

```bash
pip install -r requirements.txt
# ffmpeg must be on PATH — e.g. `sudo apt install ffmpeg` / `brew install ffmpeg`
streamlit run app.py
```

The app will load `fingerprint_db.pkl` if it exists; otherwise, it will try
to build it from a `songs/` folder placed next to `app.py` (this takes a
couple of minutes for ~50 songs and is only needed once).

## Modes

**Single-clip mode** — upload one query clip (mp3/wav/m4a/ogg/flac) and see:
- the identified song
- the top-5 candidate scores
- the clip's spectrogram
- its constellation map (the peaks used as the fingerprint)
- the offset histogram that decided the match (true match vs. runner-up)
- a side-by-side comparison of paired-hash matching vs. single-peak matching

**Batch mode** — upload several query clips (or a single `.zip` containing
them) and click "Run batch identification" to get a `results.csv` with
exactly two columns: `filename, prediction` (prediction = matched song's
filename without extension), downloadable directly from the app.

## Deploying on Streamlit Community Cloud
App URL -> https://audiofingerprintidentifier-mwzf7xbdkxskcwyv8vl6qm.streamlit.app/

github repo -> https://github.com/shrey2728/audio_fingerprint_identifier.git
## Notes

- The identification logic, hash format, and default parameters are
  identical to the Part-A notebook — nothing in `engine.py` was changed
  from what's already validated there.
- `fingerprint_db.pkl` was built with the exact same `DEFAULT_PARAMS` as
  the notebook (`sr=22050, win_length=2048, hop_length=512, ...`), so
  single-clip and batch identifications in the app reproduce the notebook's
  results exactly.
