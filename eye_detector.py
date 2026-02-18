import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np

Box = Tuple[int, int, int, int]  # x, y, w, h


@dataclass
class DetectorConfig:
    face_scale_factor: float = 1.1
    face_min_neighbors: int = 5
    face_min_size: Tuple[int, int] = (60, 60)

    eye_scale_factor: float = 1.1
    eye_min_neighbors: int = 6
    eye_min_size: Tuple[int, int] = (15, 15)

    max_eyes_per_face: int = 2
    min_eye_confidence: float = 0.35
    restrict_to_upper_face_ratio: float = 0.75


class EyeDetector:
    def __init__(self, config: DetectorConfig):
        self.cfg = config

        face_xml = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        eye_xml = os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")

        self.face_cascade = cv2.CascadeClassifier(face_xml)
        self.eye_cascade = cv2.CascadeClassifier(eye_xml)

        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load face cascade: {face_xml}")
        if self.eye_cascade.empty():
            raise RuntimeError(f"Failed to load eye cascade: {eye_xml}")

    @staticmethod
    def preprocess(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        return gray

    def detect_faces(self, gray: np.ndarray) -> List[Box]:
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.cfg.face_scale_factor,
            minNeighbors=self.cfg.face_min_neighbors,
            minSize=self.cfg.face_min_size,
        )
        return list(faces)

    def detect_eyes_in_face(self, roi_gray: np.ndarray) -> List[Box]:
        eyes = self.eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=self.cfg.eye_scale_factor,
            minNeighbors=self.cfg.eye_min_neighbors,
            minSize=self.cfg.eye_min_size,
        )
        return list(eyes)

    @staticmethod
    def eye_confidence(face_box: Box, eye_box: Box) -> float:
        _, _, fw, fh = face_box
        ex, ey, ew, eh = eye_box

        area_ratio = (ew * eh) / float(fw * fh + 1e-6)  # expected small % of face
        area_score = min(area_ratio / 0.06, 1.0)

        aspect = ew / float(eh + 1e-6)
        aspect_score = 1.0 if 0.6 <= aspect <= 2.0 else 0.0

        upper_face_score = 1.0 if ey < 0.65 * fh else 0.0

        score = 0.5 * area_score + 0.3 * upper_face_score + 0.2 * aspect_score
        return float(np.clip(score, 0.0, 1.0))

    def filter_eyes(self, face_box: Box, eye_boxes: List[Box]) -> List[Tuple[Box, float]]:
        _, _, _, fh = face_box
        candidates: List[Tuple[Box, float]] = []
        for eye in eye_boxes:
            ex, ey, ew, eh = eye

            if ey > self.cfg.restrict_to_upper_face_ratio * fh:
                continue

            conf = self.eye_confidence(face_box, eye)
            if conf < self.cfg.min_eye_confidence:
                continue

            candidates.append((eye, conf))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[: self.cfg.max_eyes_per_face]

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        gray = self.preprocess(frame)
        faces = self.detect_faces(gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 220, 40), 2)
            cv2.putText(
                frame, "Face", (x, max(0, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 220, 40), 1, cv2.LINE_AA
            )

            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            raw_eyes = self.detect_eyes_in_face(roi_gray)
            eyes = self.filter_eyes((x, y, w, h), raw_eyes)

            for (ex, ey, ew, eh), conf in eyes:
                color = (0, 255, 255) if conf >= 0.5 else (0, 140, 255)
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), color, 2)
                cv2.putText(
                    roi_color, f"Eye {conf:.2f}", (ex, max(0, ey - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
                )

        cv2.putText(
            frame, f"Faces: {len(faces)}", (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
        )
        return frame


def open_video_source(source: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    return cap


def run_image(detector: EyeDetector, image_path: str, output_path: Optional[str]) -> None:
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    out = detector.process_frame(frame)
    if output_path:
        ok = cv2.imwrite(output_path, out)
        if not ok:
            raise RuntimeError(f"Failed to write output image: {output_path}")
        print(f"Saved annotated image: {output_path}")

    cv2.imshow("Eye Detection", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_video(detector: EyeDetector, source: str, output_path: Optional[str]) -> None:
    cap = open_video_source(source)
    writer = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("End of stream or frame read failure.")
                break

            out = detector.process_frame(frame)

            if output_path and writer is None:
                h, w = out.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open output video writer: {output_path}")

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


def parse_args():
    p = argparse.ArgumentParser(description="Face-then-eye detection for images/video/webcam")
    p.add_argument("--image", type=str, default=None, help="Input image path")
    p.add_argument("--source", type=str, default="0", help="Video source index/path (default: webcam 0)")
    p.add_argument("--save", type=str, default=None, help="Save output image/video path")
    p.add_argument("--min-eye-conf", type=float, default=0.35, help="Minimum heuristic eye confidence [0,1]")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = DetectorConfig(min_eye_confidence=args.min_eye_conf)
    detector = EyeDetector(cfg)

    if args.image:
        run_image(detector, args.image, args.save)
    else:
        run_video(detector, args.source, args.save)


if __name__ == "__main__":
    main()
