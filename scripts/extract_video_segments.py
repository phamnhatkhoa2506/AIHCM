import os
import glob
import json
import zipfile
import subprocess
import pandas as pd
import cv2

def get_video_path(data_dir: str, video_id: str, temp_dir: str = "temp_videos") -> str:
    """
    Finds the video file for video_id (e.g. L21_V001).
    If stored inside Videos_*.zip, extracts it to temp_dir.
    """
    subfolder = video_id.split('_')[0]  # e.g. L21
    
    # Check if video already exists on disk
    local_candidates = [
        os.path.join(data_dir, "videos", f"{video_id}.mp4"),
        os.path.join(data_dir, f"{video_id}.mp4"),
        os.path.join(temp_dir, f"{video_id}.mp4")
    ]
    for loc in local_candidates:
        if os.path.exists(loc):
            return loc

    # Search in Videos_*.zip files
    zip_candidates = sorted(glob.glob(os.path.join(data_dir, f"Videos_{subfolder}*.zip")))
    
    for zpath in zip_candidates:
        with zipfile.ZipFile(zpath, 'r') as zf:
            internal_names = [
                f"video/{video_id}.mp4",
                f"videos/{video_id}.mp4",
                f"{video_id}.mp4"
            ]
            for int_name in internal_names:
                if int_name in zf.namelist():
                    os.makedirs(temp_dir, exist_ok=True)
                    target_path = os.path.join(temp_dir, f"{video_id}.mp4")
                    print(f"Extracting {int_name} from {os.path.basename(zpath)} -> {target_path}...")
                    with zf.open(int_name) as source, open(target_path, "wb") as target:
                        target.write(source.read())
                    return target_path

    raise FileNotFoundError(f"Could not find video {video_id}.mp4 in disk or zip files.")


def extract_video_segments_by_keyframes(
    data_dir: str,
    video_id: str,
    output_dir: str = "output_segments",
    mode: str = "between_keyframes",  # 'between_keyframes' or 'around_keyframe'
    buffer_seconds: float = 2.0,      # Only used if mode == 'around_keyframe'
    use_ffmpeg: bool = True
):
    """
    Extracts video segments mapped by map-keyframes/{video_id}.csv.
    
    Parameters:
    - data_dir: Root directory containing Competition data (map-keyframes, Videos_*.zip, etc.)
    - video_id: ID of the video (e.g. 'L21_V001')
    - output_dir: Directory to save extracted video segments
    - mode: 
        * 'between_keyframes': Segment i covers from keyframe i to keyframe i+1.
        * 'around_keyframe': Segment i covers [pts_time[i] - buffer, pts_time[i] + buffer].
    - buffer_seconds: Duration before/after keyframe when mode == 'around_keyframe'
    - use_ffmpeg: If True, uses ffmpeg CLI for fast lossless extraction. If False, uses OpenCV.
    """
    # 1. Read map-keyframes CSV
    csv_path = os.path.join(data_dir, "map-keyframes", f"{video_id}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"map-keyframes CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # Expected columns: n, pts_time, fps, frame_idx
    if 'pts_time' not in df.columns or 'n' not in df.columns:
        raise ValueError(f"CSV {csv_path} does not contain expected columns ('n', 'pts_time')")

    # 2. Get Video File Path
    video_path = get_video_path(data_dir, video_id)
    
    # Get total video duration
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = float(df['fps'].iloc[0]) if 'fps' in df.columns else 30.0
    video_duration = total_frames / fps
    cap.release()

    save_folder = os.path.join(output_dir, video_id)
    os.makedirs(save_folder, exist_ok=True)

    print(f"Processing {video_id} ({len(df)} keyframes, total duration: {video_duration:.2f}s)...")

    # 3. Extract Segments
    num_rows = len(df)
    for idx, row in df.iterrows():
        key_idx = int(row['n'])
        pts_time = float(row['pts_time'])

        if mode == "between_keyframes":
            start_time = pts_time
            if idx < num_rows - 1:
                end_time = float(df.iloc[idx + 1]['pts_time'])
            else:
                end_time = video_duration
        elif mode == "around_keyframe":
            start_time = max(0.0, pts_time - buffer_seconds)
            end_time = min(video_duration, pts_time + buffer_seconds)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        duration = max(0.1, end_time - start_time)
        out_filename = f"segment_{key_idx:03d}.mp4"
        out_filepath = os.path.join(save_folder, out_filename)

        if use_ffmpeg:
            # Fast extraction with ffmpeg
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start_time:.3f}",
                "-i", video_path,
                "-t", f"{duration:.3f}",
                "-c", "copy",
                "-loglevel", "error",
                out_filepath
            ]
            subprocess.run(cmd, check=True)
        else:
            # Fallback OpenCV extraction
            extract_segment_opencv(video_path, start_time, end_time, out_filepath, fps)

        print(f"  -> Extracted Keyframe #{key_idx:03d} ({start_time:.2f}s - {end_time:.2f}s) to {out_filename}")

    print(f"Done! All segments saved to: {save_folder}")


def extract_segment_opencv(video_path: str, start_time: float, end_time: float, out_filepath: str, fps: float):
    cap = cv2.VideoCapture(video_path)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_filepath, fourcc, fps, (width, height))

    curr_frame = start_frame
    while cap.isOpened() and curr_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        curr_frame += 1

    cap.release()
    out.release()


if __name__ == "__main__":
    DATA_DIR = r"d:\Programming\AIHCM\data\Competition"
    
    # Test extracting first 3 segments of L21_V001
    video_id = "L21_V001"
    extract_video_segments_by_keyframes(
        data_dir=DATA_DIR,
        video_id=video_id,
        output_dir="output_segments",
        mode="between_keyframes",
        use_ffmpeg=True
    )
