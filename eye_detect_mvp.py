import argparse
import os
import cv2
import numpy as np

def load_cascade(path: str, name: str) -> cv2.CascadeClassifier:
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise RuntimeError(f"Failed to load {name} cascade from: {path}")
    return cascade

def preprocess(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # local contrast boost
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)  # normalization
    return gray

def eye_confidence(face_w: int, face_h: int, ex: int, ey: int, ew: int, eh: int) -> float:
    # Heuristic confidence proxy for Haar (Haar itself does not output probabilities)
    area_ratio = (ew * eh) / float(face_w * face_h + 1e-6)
    upper_half_bonus = 1.0 if ey < 0.65 * face_h else 0.0
    aspect = ew / float(eh + 1e-6)
    aspect_bonus = 1.0 if 0.6 <= aspect <= 2.0 else 0.0
    score = 0.5 * min(area_ratio / 0.06, 1.0) + 0.3 * upper_half_bonus + 0.2 * aspect_bonus
    return float(max(0.0, min(score, 1.0)))

def detect_and_annotate(frame: np.ndarray, face_cascade, eye_cascade) -> np.ndarray:
    gray = preprocess(frame)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 220, 40), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.1, minNeighbors=6, minSize=(15, 15)
        )

        # Keep up to 2 best eyes in upper ~2/3 of face
        candidates = []
        for (ex, ey, ew, eh) in eyes:
            conf = eye_confidence(w, h, ex, ey, ew, eh)
            if ey > 0.75 * h:
                continue
            candidates.append((conf, ex, ey, ew, eh))

        candidates.sort(key=lambda t: t[0], reverse=True)
        for conf, ex, ey, ew, eh in candidates[:2]:
            color = (0, 255, 255) if conf >= 0.45 else (0, 140, 255)
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), color, 2)
            cv2.putText(
                roi_color, f"{conf:.2f}", (ex, max(0, ey - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
            )

    cv2.putText(
        frame, f"Faces: {len(faces)}", (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
    )
    return frame

def open_source(source: str):
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    return cap

def main():
    parser = argparse.ArgumentParser(description="Eye detection MVP (image or video/webcam)")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--source", type=str, default="0", help="Video source (default webcam 0)")
    parser.add_argument("--save", type=str, default=None, help="Optional output path for image/video")
    args = parser.parse_args()

    face_xml = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    eye_xml = os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")
    face_cascade = load_cascade(face_xml, "face")
    eye_cascade = load_cascade(eye_xml, "eye")

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {args.image}")
        out = detect_and_annotate(frame, face_cascade, eye_cascade)
        cv2.imshow("Eye Detection", out)
        if args.save:
            cv2.imwrite(args.save, out)
            print(f"Saved annotated image: {args.save}")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    cap = open_source(args.source)
    writer = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("End of stream or frame read failure.")
                break
            out = detect_and_annotate(frame, face_cascade, eye_cascade)

            if args.save and writer is None:
                h, w = out.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.save, fourcc, 20.0, (w, h))
            if writer is not None:
                writer.write(out)

            cv2.imshow("Eye Detection", out)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):  # ESC or q
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
