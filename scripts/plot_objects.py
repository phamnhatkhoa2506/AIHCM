import os
import json
import zipfile
import cv2
import numpy as np
from PIL import Image

def plot_objects_on_keyframe(
    data_dir: str,
    video_id: str,
    frame_id: str,
    score_threshold: float = 0.25,
    save_path: str = None
):
    """
    Plot detected objects from objects/{video_id}/{frame_id}.json onto keyframe image.
    Supports reading directly from Keyframes_*.zip without manually extracting ZIP files.
    """
    subfolder = video_id.split('_')[0] # e.g. L21
    
    zip_candidates = [
        f"Keyframes_{subfolder}.zip",
        f"Keyframes_{subfolder}_a.zip",
        f"Keyframes_{subfolder}_b.zip",
        f"Keyframes_{subfolder}_c.zip",
        f"Keyframes_{subfolder}_d.zip",
        f"Keyframes_{subfolder}_e.zip",
    ]
    
    img = None
    # 1. Read keyframe image directly from ZIP
    for zname in zip_candidates:
        zpath = os.path.join(data_dir, zname)
        if os.path.exists(zpath):
            with zipfile.ZipFile(zpath, 'r') as zf:
                img_internal_paths = [
                    f"keyframes/{video_id}/{frame_id}.jpg",
                    f"{video_id}/{frame_id}.jpg",
                    f"{frame_id}.jpg"
                ]
                for int_path in img_internal_paths:
                    if int_path in zf.namelist():
                        with zf.open(int_path) as img_file:
                            pil_img = Image.open(img_file).convert('RGB')
                            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                            break
            if img is not None:
                break

    if img is None:
        raise FileNotFoundError(f"Keyframe image for {video_id}/{frame_id} not found in zip files under {data_dir}")

    # 2. Read Object Detection JSON file
    json_path = os.path.join(data_dir, "objects", video_id, f"{frame_id}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found at: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        obj_data = json.load(f)

    scores = [float(s) for s in obj_data.get('detection_scores', [])]
    entities = obj_data.get('detection_class_entities', [])
    boxes = obj_data.get('detection_boxes', [])

    h, w, _ = img.shape

    # 3. Draw Bounding Boxes
    color_map = {}
    np.random.seed(42) # Fixed colors per class

    drawn_count = 0
    for score, entity, box in zip(scores, entities, boxes):
        if score < score_threshold:
            continue
        
        drawn_count += 1
        ymin, xmin, ymax, xmax = [float(c) for c in box]
        x1, y1 = int(xmin * w), int(ymin * h)
        x2, y2 = int(xmax * w), int(ymax * h)

        if entity not in color_map:
            color_map[entity] = tuple([int(c) for c in np.random.randint(50, 255, size=3)])
        color = color_map[entity]

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label text with background
        label = f"{entity} {score:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        cv2.rectangle(img, (x1, max(0, y1 - text_h - 4)), (x1 + text_w + 4, max(text_h + 4, y1)), color, -1)
        cv2.putText(img, label, (x1 + 2, max(text_h + 2, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    print(f"Plotted {drawn_count} objects (score >= {score_threshold}) on {video_id}/{frame_id}.jpg")

    # 4. Save visualization
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img)
        print(f"Saved result to: {save_path}")

    return img

if __name__ == "__main__":
    DATA_DIR = r"d:\Programming\AIHCM\data\Competition"
    
    # Example usage: plot objects for L21_V001 frame 001
    plot_objects_on_keyframe(
        data_dir=DATA_DIR,
        video_id="L22_V025",
        frame_id="300",
        score_threshold=0.25,
        save_path=os.path.join(DATA_DIR, "sample_objects_output.jpg")
    )
