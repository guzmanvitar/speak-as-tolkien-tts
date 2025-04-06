"""Audio preprocessing script for resampling and downmixing WAV files."""

from pydub import AudioSegment

from src.constants import DATA_PROCESSED, DATA_RECORDINGS
from src.logger_definition import get_logger

logger = get_logger(__file__)


def convert_audio():
    """Convert a stereo 48kHz WAV file to mono 16kHz WAV.

    This function reads the original Tolkien interview audio file from the
    `DATA_RECORDINGS` directory, downmixes it to mono, resamples it to 16kHz,
    and exports the result to a new file in the same directory.

    The output format is suitable for training most TTS models which expect
    16-bit PCM, mono, 16kHz WAV input.
    """
    # Paths
    input_path = DATA_RECORDINGS / "tolkien-interview.wav"
    output_path = DATA_PROCESSED / "tolkien-interview_16k.wav"

    # Load original WAV
    audio = AudioSegment.from_wav(input_path)

    # Convert to mono + 16kHz
    audio = audio.set_channels(1).set_frame_rate(16000)

    # Export
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio.export(output_path, format="wav")

    logger.info("Saved 16kHz mono WAV to %s", output_path)


if __name__ == "__main__":
    convert_audio()
