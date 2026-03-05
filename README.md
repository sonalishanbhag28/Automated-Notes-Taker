# Automated Notes Taker

An NLP-powered web application that converts lecture audio recordings into concise, structured summaries. Upload a `.wav` file and get back an automatically generated set of notes using a hybrid text summarization approach.


## How It Works

The pipeline runs in four stages:

**1. Speech-to-Text**
The uploaded audio file is transcribed using Google's Speech Recognition API via the `SpeechRecognition` library.

**2. Text Preprocessing**
The raw transcript is cleaned through a multi-step NLP pipeline:
- Contraction expansion (e.g. *don't* → *do not*) via the custom `REReplacer` class in `repRE.py`
- Word tokenization
- Lowercasing
- Punctuation and non-alphabetic character removal
- Stop word filtering
- Porter stemming

**3. Feature Extraction**
Six sentence-level features are scored to identify the most informative sentences:

| Feature | Description |
|---|---|
| F1 — Title Word Overlap | How many words from the first sentence appear in each sentence |
| F2 — Normalized Length | Sentence length relative to the longest sentence |
| F3 — Positional Score | Sentences earlier in the transcript are weighted higher |
| F4 — Numerical Data | Presence of numbers, dates, and statistics |
| F5 — Proper Nouns | Count of proper nouns detected via POS tagging |
| F6 — Similarity Matrix | Cross-sentence word overlap to detect and suppress redundancy |

**4. Sentiment-Augmented Defuzzification**
A second pass using NLTK's VADER `SentimentIntensityAnalyzer` scores each sentence by emotional polarity. Sentences are selected for the final summary using a hybrid method: those selected by both the feature scorer and the sentiment scorer are prioritized, with borderline sentences included based on a combined score threshold.

The final summary is written to `summary.txt` and displayed in the web UI.


## Project Structure

```
Automated-Notes-Taker/
│
├── app.py                  # Flask web application (main entry point)
├── notes_taker.py          # Standalone CLI version of the pipeline
├── repRE.py                # Contraction expansion utility (REReplacer class)
│
├── templates/
│   └── index.html          # Web UI — file upload form and summary display
│
├── static/
│   ├── logo.png
│   └── css/
│       ├── style.css
│       ├── nav-style.css
│       └── upload.css
│
├── nouns.wav               # Sample lecture recording — Nouns (Lecture 11)
├── adjts.wav               # Sample lecture recording — Adjectives (Lecture 12)
├── verbs.wav               # Sample lecture recording — Verbs (Lecture 13)
│
├── nouns.txt               # Transcript output for nouns lecture
├── adjts.txt               # Transcript output for adjectives lecture
├── verbs.txt               # Transcript output for verbs lecture
│
├── converted_text.txt      # Raw speech-to-text output (auto-generated)
├── input.txt               # Preprocessed text (auto-generated)
└── summary.txt             # Final summary output (auto-generated)
```


## Requirements

Install dependencies with:

```bash
pip install flask werkzeug SpeechRecognition pydub textblob nltk pandas tabulate
```

You will also need **FFmpeg** installed on your system for `pydub` to handle audio files.

On first run, the following NLTK data packages are downloaded automatically:
- `stopwords`
- `averaged_perceptron_tagger`
- `vader_lexicon`
- `punkt`


## Setup & Usage

### Web Application

1. Load the contraction-expansion module:
   ```bash
   python repRE.py
   ```

2. Start the Flask server:
   ```bash
   flask run
   ```

3. Open your browser and navigate to `http://127.0.0.1:5000/`

4. Upload a `.wav` audio file and click **Upload**. The generated summary will appear on the page.

### CLI Version

The standalone script `notes_taker.py` runs a menu-driven version of the same pipeline against the three bundled sample recordings:

```bash
python notes_taker.py
```

You will be prompted to select a lecture number (11, 12, or 13) and can choose between displaying a pre-generated summary or re-running the speech-to-text conversion from scratch.


## Configuration

The summary length and sensitivity can be tuned by editing two thresholds in `app.py` (and equivalently in `notes_taker.py`):

```python
# Minimum combined feature score for a sentence to be included
if score[i] >= 3.0:
    ...

# Sentences shorter than 40% of the longest sentence are excluded
if f2[i] < 0.4:
    ...

# Sentences with >80% word overlap with another are treated as duplicates
if f6[i] > 0.8:
    ...
```

Raising the `score[i]` threshold produces shorter, more selective summaries. Lowering it includes more sentences.


## Notes

- Speech recognition accuracy depends on audio quality, Python version, and the local speech recognizer configuration. Results may vary across environments.
- The application currently expects `.wav` format audio input.
- The `notes_taker.py` CLI is scoped to the three bundled Foundation English lecture recordings. The web app (`app.py`) accepts any uploaded audio file.
