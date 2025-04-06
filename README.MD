# Speak as Tolkien - Custom TTS
This project aims to synthesize audio from text using TTS models using style transfer to bring Tolkien's characteristic voice to life, capturing his rhythm, intonations, stuttering, and elongations as much as possible.

## Features

- 🧠 Style-aware TTS model (custom trained or cloned)
- 🗣️ Focus on Tolkien's voice and speech quirks
- 🔌 API to input text and retrieve audio files
- 🛠️ Modular architecture for training, inference, and serving

##  Tech Stack

- Coqui TTS or VITS (local training)
- FastAPI for the API layer
- Audio preprocessing with Librosa or Torchaudio

## 🔧 Setup & Installation
1️⃣ Clone the Repository
```bash
git clone https://github.com/guzmanvitar/dungeonmind-llm-rpg-adventure.git
cd dungeonmind-llm-rpg-adventure
```
2️⃣ Install Dependencies
This repo uses [uv](https://docs.astral.sh/uv/getting-started/installation) to install and manage dependencies,
as well as to set up the Python environment. After installing `uv` run
```bash
uv python install 3.9.18
uv sync
```
To set up Git hooks for code quality checks run also
```bash
uv run pre-commit install
```
