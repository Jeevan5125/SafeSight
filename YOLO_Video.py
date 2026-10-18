from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
WEIGHTS = PROJECT_DIR / "YOLO-Weights" / "ppe.pt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIOLATION_CLASSES = {"no-hardhat", "no-mask", "no-safety vest"}
EQUIPMENT_CLASSES = {"machinery", "vehicle"}
CONFIDENCE_THRESHOLD = 0.5


def _class_name(names, class_id):
    """Return a class label for either Ultralytics list or dict name mappings."""
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), class_id)))
    return str(names[class_id])


def _annotate(img, results, names):
    """Draw detections and a frame-level PPE alert summary on the image."""
    violations = 0
    equipment = 0

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            class_id = int(box.cls[0])
            class_name = _class_name(names, class_id)
            normalized_name = class_name.strip().lower()
            if normalized_name in VIOLATION_CLASSES:
                violations += 1
                color = (0, 0, 255)
            elif normalized_name in {"hardhat", "mask", "safety vest"}:
                color = (0, 200, 90)
            elif normalized_name in EQUIPMENT_CLASSES or normalized_name == "safety cone":
                equipment += 1
                color = (0, 149, 255)
            else:
                color = (220, 90, 190)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{class_name} {confidence:.2f}"
            text_width, text_height = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
            )[0]
            label_top = max(y1, text_height + 10)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(
                img,
                (x1, label_top - text_height - 8),
                (x1 + text_width + 8, label_top),
                color,
                -1,
            )
            cv2.putText(
                img,
                label,
                (x1 + 4, label_top - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    # This is a per-frame alert heuristic, not a unique-person compliance score.
    if violations >= 2:
        status = "HIGH ALERT"
        banner_color = (35, 35, 220)
    elif violations == 1:
        status = "PPE ALERT"
        banner_color = (0, 125, 230)
    else:
        status = "NO VIOLATION CLASS DETECTED"
        banner_color = (45, 125, 60)

    banner_height = 42
    cv2.rectangle(img, (0, 0), (img.shape[1], banner_height), (20, 27, 30), -1)
    summary = f"{status}  |  PPE alerts: {violations}  |  Equipment: {equipment}"
    cv2.putText(
        img,
        summary,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        banner_color,
        2,
        cv2.LINE_AA,
    )
    return img


def _open_camera(camera_index):
    """Open a local webcam using DirectShow first on Windows, then auto backend."""
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_index)
    return cap


def video_detection(path_x):
    if path_x is None:
        return

    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {WEIGHTS}")

    model = YOLO(str(WEIGHTS))
    names = model.names

    # Accept integer 0 or text values sent by a Flask route for the default camera.
    is_webcam = (isinstance(path_x, int) and path_x >= 0) or str(path_x).strip().lower() in {
        "0", "webcam", "camera"
    }

    if is_webcam:
        camera_index = path_x if isinstance(path_x, int) else 0
        cap = _open_camera(camera_index)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"Could not open webcam {camera_index}. Close other camera apps and check Windows camera permissions."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        try:
            while True:
                success, img = cap.read()
                if not success or img is None:
                    break
                results = model(img, verbose=False)
                yield _annotate(img, results, names)
        finally:
            cap.release()
        return

    input_path = Path(path_x)
    if input_path.suffix.lower() in IMAGE_EXTENSIONS:
        img = cv2.imread(str(input_path))
        if img is None:
            raise ValueError(f"Could not read image: {input_path}")
        results = model(img, verbose=False)
        annotated = _annotate(img, results, names)
        output_path = input_path.with_name(f"{input_path.stem}_output.jpg")
        if not cv2.imwrite(str(output_path), annotated):
            raise OSError(f"Could not save annotated image: {output_path}")
        yield annotated
        return

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Could not open video: {input_path}")

    try:
        while True:
            success, img = cap.read()
            if not success or img is None:
                break
            results = model(img, verbose=False)
            yield _annotate(img, results, names)
    finally:
        cap.release()
