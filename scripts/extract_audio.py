import os
import glob
import subprocess
from concurrent.futures import ThreadPoolExecutor

def extract_audio_from_video(video_path: str, output_audio_path: str, sample_rate: int = 16000) -> bool:
    """
    Extracts audio from a video file into 16kHz Mono WAV format (ideal for Whisper ASR).
    
    Parameters:
    - video_path: Path to input .mp4 file
    - output_audio_path: Path to output .wav file
    - sample_rate: Audio sampling rate (default 16000Hz for ASR)
    """
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)

    # ffmpeg command: extract mono audio at 16kHz WAV format
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                      # Disable video recording
        "-acodec", "pcm_s16le",      # Uncompressed 16-bit PCM WAV
        "-ar", str(sample_rate),    # Sampling rate (16000 Hz)
        "-ac", "1",                 # Mono channel
        "-loglevel", "error",
        output_audio_path
    ]

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def batch_extract_audio_segments(
    segments_dir: str = "output_segments",
    output_audio_dir: str = "output_audios",
    video_id: str = None,
    num_workers: int = 4
):
    """
    Batch extracts audio for all segment .mp4 files.
    
    Parameters:
    - segments_dir: Parent folder containing extracted video segment folders (e.g. output_segments)
    - output_audio_dir: Destination folder for .wav files
    - video_id: If specified (e.g. 'L21_V001'), only extracts for that video. If None, processes all subfolders.
    - num_workers: Number of threads for parallel processing
    """
    if video_id:
        video_dirs = [os.path.join(segments_dir, video_id)]
    else:
        video_dirs = [d for d in glob.glob(os.path.join(segments_dir, "*")) if os.path.isdir(d)]

    if not video_dirs:
        print(f"No video segment directories found in '{segments_dir}'!")
        return

    for v_dir in video_dirs:
        vid_name = os.path.basename(v_dir)
        mp4_files = sorted(glob.glob(os.path.join(v_dir, "*.mp4")))
        
        if not mp4_files:
            print(f"No .mp4 files found in {v_dir}")
            continue

        target_out_dir = os.path.join(output_audio_dir, vid_name)
        os.makedirs(target_out_dir, exist_ok=True)

        print(f"Processing audio extraction for {vid_name} ({len(mp4_files)} segments)...")

        tasks = []
        for mp4_path in mp4_files:
            base_name = os.path.splitext(os.path.basename(mp4_path))[0]
            out_wav = os.path.join(target_out_dir, f"{base_name}.wav")
            tasks.append((mp4_path, out_wav))

        # Parallel extraction
        success_count = 0
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(lambda t: extract_audio_from_video(t[0], t[1]), tasks))
            success_count = sum(1 for r in results if r)

        print(f"-> Successfully extracted {success_count}/{len(mp4_files)} audio files to '{target_out_dir}'")


if __name__ == "__main__":
    # Example: Extract audio for L21_V001 video segments
    batch_extract_audio_segments(
        segments_dir="output_segments",
        output_audio_dir="output_audios",
        video_id="L21_V001",
        num_workers=4
    )
