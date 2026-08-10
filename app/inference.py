import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms
from collections import deque
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import EmotionNet
from src.preprocess import apply_clahe


EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

EMOTION_COLORS = {
    'angry'   : (0,   0,   255),
    'disgust' : (0,   140, 255),
    'fear'    : (128, 0,   128),
    'happy'   : (0,   255, 0  ),
    'neutral' : (255, 255, 0  ),
    'sad'     : (255, 0,   0  ),
    'surprise': (0,   255, 255),
}

# FIX 2 — minimum confidence to show emotion label
MIN_CONFIDENCE = 0.45

# FIX 3 — only run model every N frames
PREDICT_EVERY_N_FRAMES = 3


# FIX 1 — temporal smoother class
class EmotionSmoother:
    """
    Averages emotion scores across last window_size frames.
    Stabilizes predictions against single-frame noise.
    window_size=10 means average last 10 frames (~0.5s at 20fps)
    """
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.history     = deque(maxlen=window_size)

    def update(self, scores):
        """Add new frame scores and return smoothed result."""
        self.history.append(scores)
        return self.get_smoothed()

    def get_smoothed(self):
        """Average all scores in history window."""
        if not self.history:
            return {e: 0.0 for e in EMOTIONS}
        smoothed = {e: 0.0 for e in EMOTIONS}
        for frame_scores in self.history:
            for e, s in frame_scores.items():
                smoothed[e] += s
        n = len(self.history)
        return {e: v / n for e, v in smoothed.items()}

    def reset(self):
        self.history.clear()


def load_model(model_path, device):
    model = EmotionNet(num_classes=7, dropout=0.0)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    print(f'Model loaded  : {model_path}')
    return model


def get_inference_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(apply_clahe),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


def predict_emotion(face_crop, model, transform, device):
    image = transform(face_crop)
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probs   = torch.softmax(outputs, dim=1)[0]
    scores = {EMOTIONS[i]: float(probs[i]) for i in range(len(EMOTIONS))}
    top_emotion = max(scores, key=scores.get)
    return top_emotion, scores


def draw_results(frame, boxes, emotions_list):
    """Draw bounding box and emotion label — only if confidence above threshold."""
    if boxes is None:
        return frame
    for box, (emotion, scores) in zip(boxes, emotions_list):
        x1, y1, x2, y2 = [int(b) for b in box]
        color      = EMOTION_COLORS.get(emotion, (255, 255, 255))
        confidence = scores[emotion]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # FIX 2 — only show label if confidence above threshold
        if confidence >= MIN_CONFIDENCE:
            label = f'{emotion} {confidence:.0%}'
            text_color = color
        else:
            label = 'uncertain'
            text_color = (128, 128, 128)

        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    return frame


def count_emotions(emotions_list):
    """Count faces per emotion and calculate ratio."""
    counts = {e: 0 for e in EMOTIONS}
    total  = len(emotions_list)
    for emotion, scores in emotions_list:
        # FIX 2 — only count if confidence above threshold
        if scores[emotion] >= MIN_CONFIDENCE:
            counts[emotion] += 1
    ratios = {e: counts[e] / total if total > 0 else 0.0 for e in EMOTIONS}
    return counts, ratios, total


def draw_mood_panel(frame, counts, ratios, total, fps, smoothing_window):
    """Draw mood panel on right side of frame."""
    h, w      = frame.shape[:2]
    panel_w   = 230
    panel     = np.zeros((h, panel_w, 3), dtype=np.uint8)
    panel[:]  = (30, 30, 30)

    y = 20

    cv2.putText(panel, 'AudiMood', (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y += 25

    cv2.putText(panel, f'FPS: {fps:.1f}', (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    y += 18

    # FIX 1 — show smoothing window size
    cv2.putText(panel, f'Smooth: {smoothing_window} frames', (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    y += 18

    # FIX 2 — show confidence threshold
    cv2.putText(panel, f'Min conf: {MIN_CONFIDENCE:.0%}', (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    y += 18

    cv2.putText(panel, f'Total faces: {total}', (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    y += 25

    cv2.line(panel, (10, y), (panel_w - 10, y), (80, 80, 80), 1)
    y += 15

    cv2.putText(panel, 'Emotion breakdown', (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    y += 20

    bar_max_w = panel_w - 100

    for emotion in EMOTIONS:
        count = counts[emotion]
        ratio = ratios[emotion]
        color = EMOTION_COLORS[emotion]
        alpha = 255 if count > 0 else 100

        label = f'{emotion[:7]:7s} {count:2d}'
        cv2.putText(panel, label, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (alpha, alpha, alpha), 1)

        bar_x = 110
        bar_h = 8
        cv2.rectangle(panel, (bar_x, y - 8),
                      (bar_x + bar_max_w, y), (60, 60, 60), -1)

        fill_w = int(bar_max_w * ratio)
        if fill_w > 0:
            cv2.rectangle(panel, (bar_x, y - 8),
                          (bar_x + fill_w, y), color, -1)

        pct_text = f'{ratio:.0%}'
        cv2.putText(panel, pct_text,
                    (bar_x + bar_max_w + 3, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (alpha, alpha, alpha), 1)
        y += 22

    y += 5
    cv2.line(panel, (10, y), (panel_w - 10, y), (80, 80, 80), 1)
    y += 15

    if total > 0:
        dominant  = max(counts, key=counts.get)
        dom_color = EMOTION_COLORS[dominant]
        cv2.putText(panel, 'Dominant:', (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        y += 18
        cv2.putText(panel, f'{dominant.upper()} {ratios[dominant]:.0%}',
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, dom_color, 2)

    combined = np.hstack([frame, panel])
    return combined


def aggregate_emotions(emotions_list):
    if not emotions_list:
        return {e: 0.0 for e in EMOTIONS}
    crowd = {e: 0.0 for e in EMOTIONS}
    for _, scores in emotions_list:
        for emotion, score in scores.items():
            crowd[emotion] += score
    n = len(emotions_list)
    return {e: v / n for e, v in crowd.items()}


def run_inference(
    model_path,
    source          = 'webcam',
    device_name     = 'cuda',
    show_window     = True,
    max_faces       = 20,
    smoothing_window = 10,    # FIX 1 — smoothing window size
):
    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
    print(f'Inference on  : {device}')
    print(f'Smooth window : {smoothing_window} frames')
    print(f'Min confidence: {MIN_CONFIDENCE:.0%}')
    print(f'Predict every : {PREDICT_EVERY_N_FRAMES} frames')

    model     = load_model(model_path, device)
    mtcnn     = MTCNN(keep_all=True, device=device, min_face_size=20)
    transform = get_inference_transform()

    if source == 'webcam':
        cap = cv2.VideoCapture(0)
        print('Opening webcam...')
    else:
        cap = cv2.VideoCapture(source)
        print(f'Opening video : {source}')

    if not cap.isOpened():
        print('ERROR: could not open video source')
        return

    print('Press Q to quit')

    # FIX 1 — one smoother per face slot (track up to max_faces)
    smoothers = [EmotionSmoother(window_size=smoothing_window)
                 for _ in range(max_faces)]

    prev_time     = time.time()
    fps           = 0
    frame_count   = 0
    last_emotions = []   # FIX 3 — cache last prediction
    last_boxes    = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print('End of video or camera disconnected')
            break

        frame_count += 1

        # FIX 3 — only run detection every N frames
        if frame_count % PREDICT_EVERY_N_FRAMES == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(rgb_frame)

            boxes, _ = mtcnn.detect(pil_frame)
            emotions_list = []

            if boxes is not None:
                boxes = boxes[:max_faces]
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = [int(b) for b in box]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)

                    if (x2 - x1) < 20 or (y2 - y1) < 20:
                        continue

                    face_crop = pil_frame.crop((x1, y1, x2, y2))
                    emotion, scores = predict_emotion(
                        face_crop, model, transform, device
                    )

                    # FIX 1 — smooth scores through history window
                    smoothed_scores = smoothers[i].update(scores)
                    smoothed_emotion = max(smoothed_scores, key=smoothed_scores.get)
                    emotions_list.append((smoothed_emotion, smoothed_scores))

            # cache results for frames we skip
            last_emotions = emotions_list
            last_boxes    = boxes
        else:
            # FIX 3 — reuse last prediction on skipped frames
            emotions_list = last_emotions
            boxes         = last_boxes

        # calculate FPS
        curr_time = time.time()
        fps       = 1 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        # draw
        if boxes is not None and len(emotions_list) > 0:
            frame = draw_results(frame, boxes[:len(emotions_list)], emotions_list)

        counts, ratios, total = count_emotions(emotions_list)
        frame = draw_mood_panel(frame, counts, ratios, total, fps, smoothing_window)

        if show_window:
            cv2.imshow('AudiMood — Emotion Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Stopped by user')
                break

    cap.release()
    cv2.destroyAllWindows()
    print('Inference stopped')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='AudiMood Inference')
    parser.add_argument('--model_path',       type=str,
                        default='models/best_optimized_dropout05_es.pth')
    parser.add_argument('--source',           type=str, default='webcam')
    parser.add_argument('--device',           type=str, default='cuda')
    parser.add_argument('--max_faces',        type=int, default=20)
    parser.add_argument('--smoothing_window', type=int, default=10)
    parser.add_argument('--no_window',        action='store_true')
    args = parser.parse_args()

    run_inference(
        model_path       = args.model_path,
        source           = args.source,
        device_name      = args.device,
        show_window      = not args.no_window,
        max_faces        = args.max_faces,
        smoothing_window = args.smoothing_window,
    )