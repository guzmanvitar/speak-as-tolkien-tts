"""Segment audio based on Whisper transcription."""

import csv
from pathlib import Path

import whisper
from pydub import AudioSegment

from src.constants import DATA_PROCESSED
from src.logger_definition import get_logger

logger = get_logger(__file__)


def transcribe_audio(audio_path: Path, model_size: str = "base") -> list[dict]:
    """Transcribe audio using OpenAI Whisper.

    Args:
        audio_path (Path): Path to the .wav file to transcribe.
        model_size (str): Whisper model size (tiny, base, small, medium, large).

    Returns:
        List[dict]: A list of transcription segments with start, end, and text.
    """
    logger.info("Loading Whisper model: %s", model_size)
    model = whisper.load_model(model_size)

    logger.info("Transcribing audio: %s", audio_path)
    result = model.transcribe(
        str(audio_path),
        best_of=5,
        initial_prompt="The speaker speaks slowly and carefully. Wait for long silences to"
        " segment.",
    )
    segments = result.get("segments", [])
    logger.info("Transcription complete. Found %d segments.", len(segments))
    return segments


def segment_and_save_audio(
    audio_path: Path, segments: list[dict], output_dir: Path
) -> None:
    """Slice audio into chunks based on Whisper segments and save them.

    Args:
        audio_path (Path): Path to original audio file.
        segments (List[dict]): List of segments from Whisper.
        output_dir (Path): Directory to save segmented .wav files and CSV.
    """
    logger.info("Loading audio for slicing: %s", audio_path)
    audio = AudioSegment.from_wav(audio_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "segments.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "start", "end", "text"])

        for i, segment in enumerate(segments):
            start_ms = int(segment["start"] * 1000)
            end_ms = int(segment["end"] * 1000)
            clip = audio[start_ms:end_ms]

            filename = f"segment_{i:03}.wav"
            clip.export(output_dir / filename, format="wav")

            writer.writerow(
                [filename, segment["start"], segment["end"], segment["text"]]
            )
            logger.debug(
                "Saved segment %s (%.2f → %.2f)",
                filename,
                segment["start"],
                segment["end"],
            )

    logger.info("Saved all audio segments and CSV to %s", output_dir)


def main() -> None:
    """Run transcription and segmentation for the Tolkien audio."""
    input_audio = DATA_PROCESSED / "tolkien-interview_16k.wav"
    output_dir = DATA_PROCESSED / "chunks"

    segments = transcribe_audio(input_audio, model_size="base")
    segment_and_save_audio(input_audio, segments, output_dir)


if __name__ == "__main__":
    main()
