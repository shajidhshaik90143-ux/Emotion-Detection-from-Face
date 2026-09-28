import cv2
import numpy as np

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

def _normalize_face_region(region):
    return {
        "x": int(region.get("x", 0)),
        "y": int(region.get("y", 0)),
        "w": int(region.get("w", 0)),
        "h": int(region.get("h", 0)),
    }

def analyze_image(frame_bgr, DeepFace, detector_backend="opencv"):
    """
    Analyze all detected faces using DeepFace.
    DeepFace returns emotion probabilities; the highest probability is used.
    """
    try:
        raw = DeepFace.analyze(
            img_path=frame_bgr,
            actions=["emotion"],
            detector_backend=detector_backend,
            enforce_detection=False,
            align=True,
            silent=True,
        )
    except TypeError:
        # Compatibility fallback for DeepFace versions without silent.
        raw = DeepFace.analyze(
            img_path=frame_bgr,
            actions=["emotion"],
            detector_backend=detector_backend,
            enforce_detection=False,
            align=True,
        )

    if isinstance(raw, dict):
        raw = [raw]

    results = []
    for idx, item in enumerate(raw, start=1):
        emotion_scores = item.get("emotion", {}) or {}
        if emotion_scores:
            emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = float(emotion_scores[emotion])
        else:
            emotion = str(item.get("dominant_emotion", "unknown"))
            confidence = 0.0

        region = _normalize_face_region(item.get("region", {}))
        results.append({
            "face_index": idx,
            "emotion": emotion.lower(),
            "confidence": confidence,
            "scores": {k: float(v) for k, v in emotion_scores.items()},
            "region": region,
        })

    return results

def draw_results(frame, results, show_box=True, show_label=True):
    for r in results:
        region = r["region"]
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]

        # Clamp coordinates to the frame.
        h_img, w_img = frame.shape[:2]
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = max(1, min(w, w_img - x))
        h = max(1, min(h, h_img - y))

        if show_box:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (40, 220, 120), 2)

        if show_label:
            text = f"Face {r['face_index']}: {r['emotion'].title()} {r['confidence']:.1f}%"
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .62, 2)
            top = max(0, y - th - baseline - 8)
            cv2.rectangle(frame, (x, top), (min(w_img, x+tw+12), y), (40, 220, 120), -1)
            cv2.putText(frame, text, (x+6, y-7), cv2.FONT_HERSHEY_SIMPLEX, .62, (0, 0, 0), 2, cv2.LINE_AA)

    return frame
