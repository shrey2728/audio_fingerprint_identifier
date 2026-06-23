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

1. Create a new GitHub repo and push these files:
   ```bash
   git init
   git add app.py engine.py fingerprint_db.pkl requirements.txt packages.txt README.md
   git commit -m "EE200 Q3-B fingerprinting app"
   git branch -M main
   git remote add origin <your-repo-url>
   git push -u origin main
   ```
   **Do not push the `songs/` folder** — it's ~370 MB of mp3s and is not
   needed at runtime (the app only needs `fingerprint_db.pkl`, ~49 MB,
   which is under GitHub's 100 MB per-file limit). If you want to keep the
   audio files in the repo for completeness, use [Git LFS](https://git-lfs.com/)
   or a `.gitignore` entry for `songs/` and host them separately.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, and click "New app".
3. Pick your repo, branch `main`, and main file path `app.py`.
4. Click "Deploy". Streamlit Cloud will read `requirements.txt` (Python
   packages) and `packages.txt` (system packages — installs `ffmpeg`)
   automatically, then start the app. First boot takes a minute or two
   while it loads `fingerprint_db.pkl`.
5. Once live, the URL it gives you (`https://<your-app>.streamlit.app`) is
   what you submit.

## Notes

- The identification logic, hash format, and default parameters are
  identical to the Part-A notebook — nothing in `engine.py` was changed
  from what's already validated there.
- `fingerprint_db.pkl` was built with the exact same `DEFAULT_PARAMS` as
  the notebook (`sr=22050, win_length=2048, hop_length=512, ...`), so
  single-clip and batch identifications in the app reproduce the notebook's
  results exactly.
