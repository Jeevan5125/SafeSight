"""Zone-based, track-persistent PPE violation events for the local demo."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "YOLO-Weights" / "ppe.pt"
DATA_DIR = ROOT / "safety_data"
EVIDENCE_DIR = ROOT / "static" / "evidence"
ZONES_FILE = DATA_DIR / "zones.json"
EVENTS_DB = DATA_DIR / "events.sqlite3"
SETTINGS_FILE = DATA_DIR / "site_settings.json"
PERSIST_SECONDS = 1.5
LOST_TRACK_SECONDS = 3.0
# Bridge brief missed detections and the low FPS of local CPU inference while
# keeping a genuine zone exit from carrying dwell time indefinitely.
ZONE_EXIT_GRACE_SECONDS = 2.0
CONFIDENCE = 0.40

VIOLATIONS = {
    "no-hardhat": "Hardhat missing",
    "no-mask": "Mask missing",
    "no-safety vest": "Safety vest missing",
    "zone-dwell": "Person stayed in zone for over 1.5 seconds",
}
PPE_CLASSES = {
    "hardhat": ("Hardhat", "NO-Hardhat", 0.14),
    "mask": ("Mask", "NO-Mask", 0.30),
    "safety vest": ("Safety Vest", "NO-Safety Vest", 0.58),
}
PPE_CLASS_LOOKUP = {
    label.lower(): (item, label == pair[0])
    for item, pair in PPE_CLASSES.items()
    for label in pair[:2]
}
RULE_TO_CLASS = {
    "Hardhat": "no-hardhat",
    "Mask": "no-mask",
    "Safety Vest": "no-safety vest",
}
ALLOWED_RULES = set(RULE_TO_CLASS)


def _connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(EVENTS_DB, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            occurred_at TEXT NOT NULL,
            zone_name TEXT NOT NULL,
            track_id INTEGER NOT NULL,
            violation TEXT NOT NULL,
            confidence REAL NOT NULL,
            evidence_file TEXT NOT NULL,
            ppe_status TEXT NOT NULL DEFAULT ''
        )"""
    )
    columns = {row[1] for row in connection.execute("PRAGMA table_info(events)")}
    if "ppe_status" not in columns:
        connection.execute("ALTER TABLE events ADD COLUMN ppe_status TEXT NOT NULL DEFAULT ''")
    connection.commit()
    return connection


def list_events(limit=50):
    connection = _connect()
    try:
        rows = connection.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


class SafetyMonitor:
    def __init__(self):
        if not WEIGHTS.is_file():
            raise FileNotFoundError(f"YOLO checkpoint not found: {WEIGHTS}")
        self.model = YOLO(str(WEIGHTS))
        self._lock = threading.RLock()
        # Dashboard and zone-editor may both request the camera feed. Serialize
        # full frame processing so tracking and persistent zone state stay coherent.
        self._process_lock = threading.Lock()
        self._zones = self._load_zones()
        self._settings = self._load_settings()
        self._snapshot = {
            "updated_at": None,
            "tracked_people": 0,
            "zone_people": [],
            "class_counts": {},
            "machinery_count": 0,
            "ppe_alert_count": 0,
            "site_closed": False,
            "closing_time": self._settings["closing_time"],
            "after_hours_people": 0,
        }
        self._after_hours_alerted = set()
        # Key -> {since, last_seen, logged}; retained across streamed frames.
        self._active = {}
        self._seen_track_ids = set()

    def _load_zones(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not ZONES_FILE.exists():
            return []
        try:
            value = json.loads(ZONES_FILE.read_text(encoding="utf-8"))
            return value if isinstance(value, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    def zones(self):
        with self._lock:
            return json.loads(json.dumps(self._zones))

    def _load_settings(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        defaults = {"closing_time": ""}
        try:
            stored = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            closing_time = str(stored.get("closing_time", "")).strip()
            if closing_time:
                datetime.strptime(closing_time, "%H:%M")
            return {"closing_time": closing_time}
        except (OSError, json.JSONDecodeError, AttributeError, ValueError):
            return defaults

    def settings(self):
        with self._lock:
            return dict(self._settings)

    def update_settings(self, payload):
        closing_time = str(payload.get("closing_time", "")).strip()
        if closing_time:
            try:
                datetime.strptime(closing_time, "%H:%M")
            except ValueError as exc:
                raise ValueError("Closing time must use 24-hour HH:MM format.") from exc
        with self._lock:
            self._settings = {"closing_time": closing_time}
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            temporary = SETTINGS_FILE.with_suffix(".tmp")
            temporary.write_text(json.dumps(self._settings, indent=2), encoding="utf-8")
            temporary.replace(SETTINGS_FILE)
            return dict(self._settings)

    def snapshot(self):
        with self._lock:
            return json.loads(json.dumps(self._snapshot))

    def add_zone(self, payload):
        name = str(payload.get("name", "")).strip()[:40]
        points = payload.get("points")
        required = payload.get("required", [])
        if not name:
            raise ValueError("Give the zone a name.")
        if not isinstance(points, list) or len(points) < 3 or len(points) > 30:
            raise ValueError("A zone needs 3 to 30 polygon points.")
        clean_points = []
        for point in points:
            x, y = float(point["x"]), float(point["y"])
            if not (0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError("Zone points must be normalized from 0 to 1.")
            clean_points.append({"x": x, "y": y})
        if not isinstance(required, list) or not required:
            raise ValueError("Select at least one required PPE item.")
        clean_required = sorted(set(required) & ALLOWED_RULES)
        if not clean_required:
            raise ValueError("Selected PPE rules are not recognized.")

        with self._lock:
            zone = {
                "id": uuid.uuid4().hex[:10],
                "name": name,
                "points": clean_points,
                "required": clean_required,
            }
            self._zones.append(zone)
            self._save_zones()
            return zone

    def remove_zone(self, zone_id):
        with self._lock:
            old_count = len(self._zones)
            self._zones = [zone for zone in self._zones if zone["id"] != zone_id]
            if len(self._zones) != old_count:
                self._save_zones()

    def _save_zones(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        temporary = ZONES_FILE.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._zones, indent=2), encoding="utf-8")
        temporary.replace(ZONES_FILE)

    @staticmethod
    def _name(class_names, class_id):
        if isinstance(class_names, dict):
            return str(class_names.get(class_id, class_names.get(str(class_id), class_id)))
        return str(class_names[class_id])

    @staticmethod
    def _inside(point, polygon):
        contour = np.array(polygon, dtype=np.int32)
        return cv2.pointPolygonTest(contour, point, False) >= 0

    @classmethod
    def _person_in_zone(cls, box, polygon):
        """Use lower-body anchors so cropped or occluded feet don't miss a zone."""
        x1, y1, x2, y2 = box
        box_height = max(1, y2 - y1)
        center_x = (x1 + x2) // 2
        # The foot point is the most precise anchor. The lower-leg anchors
        # handle camera crops and partial occlusion when the feet aren't visible.
        anchors = [
            (center_x, max(y1, y2 - 2)),
            (center_x, int(y1 + 0.90 * box_height)),
            (center_x, int(y1 + 0.78 * box_height)),
        ]
        return any(cls._inside(point, polygon) for point in anchors)

    @staticmethod
    def _ppe_status(votes, required):
        statuses = []
        for rule in required:
            positive, negative, _ = PPE_CLASSES[rule.lower()]
            scores = votes.get(rule, {})
            yes = scores.get(positive.lower(), 0.0)
            no = scores.get(negative.lower(), 0.0)
            if not yes and not no:
                label = "Not detected / uncertain"
            elif yes and no and abs(yes - no) <= 0.2 * max(yes, no):
                label = "Uncertain"
            else:
                label = "Present" if yes > no else "Missing"
            statuses.append((rule, label))

        missing = [name for name, state in statuses if state == "Missing"]
        if missing:
            summary = "PPE violation: missing " + ", ".join(missing)
        elif statuses and all(state == "Present" for _, state in statuses):
            summary = "PPE complete"
        else:
            summary = "PPE status uncertain"
        detail = "; ".join(f"{name}: {state}" for name, state in statuses)
        return summary, detail

    def _record_event(self, zone, track_id, violation, confidence, frame, ppe_status="", ppe_summary=""):
        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        now = datetime.now().astimezone()
        filename = f"event_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg"
        evidence_path = EVIDENCE_DIR / filename
        if not cv2.imwrite(str(evidence_path), frame):
            return
        connection = _connect()
        try:
            connection.execute(
                """INSERT INTO events
                   (occurred_at, zone_name, track_id, violation, confidence, evidence_file, ppe_status)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    now.isoformat(timespec="seconds"),
                    zone["name"],
                    int(track_id),
                    (
                        f"Zone dwell — {ppe_summary}"
                        if violation == "zone-dwell"
                        else f"After-hours person detected — {ppe_summary}"
                        if violation == "after-hours"
                        else VIOLATIONS.get(violation, violation)
                    ),
                    float(confidence),
                    filename,
                    ppe_status,
                ),
            )
            connection.commit()
        finally:
            connection.close()

    def process(self, frame):
        with self._process_lock:
            return self._process_frame(frame)

    def _process_frame(self, frame):
        with self._lock:
            zones = json.loads(json.dumps(self._zones))
            settings = dict(self._settings)

        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONFIDENCE,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes
        names = self.model.names
        height, width = frame.shape[:2]
        people = []
        ppe_boxes = []
        frame_counts = {}
        active_now = set()
        ppe_alerts = set()
        occupants = []
        now = time.monotonic()

        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0])
                name = self._name(names, class_id).strip()
                normalized = name.lower()
                frame_counts[normalized] = frame_counts.get(normalized, 0) + 1
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0]) if box.id is not None else None
                color = (45, 195, 100)
                if normalized == "person":
                    if track_id is not None:
                        self._seen_track_ids.add(track_id)
                    people.append({"id": track_id, "box": (x1, y1, x2, y2), "confidence": confidence})
                    color = (220, 150, 25)
                elif normalized in PPE_CLASS_LOOKUP:
                    ppe_boxes.append(
                        {"name": normalized, "box": (x1, y1, x2, y2), "confidence": confidence}
                    )
                    color = (45, 195, 100) if PPE_CLASS_LOOKUP[normalized][1] else (35, 35, 235)
                elif normalized in {"machinery", "vehicle", "safety cone"}:
                    color = (0, 149, 255)

                label = f"{name} {confidence:.2f}"
                if normalized == "person" and track_id is not None:
                    label = f"Person #{track_id} {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        # Link each PPE detection to at most one tracked person. The normalized
        # vertical position helps distinguish a nearby worker's helmet/vest when
        # person boxes overlap in a crowded scene.
        for person in people:
            person["ppe_observations"] = {}
        for detected in ppe_boxes:
            item, _is_present = PPE_CLASS_LOOKUP[detected["name"]]
            vx1, vy1, vx2, vy2 = detected["box"]
            center_x, center_y = (vx1 + vx2) // 2, (vy1 + vy2) // 2
            candidates = []
            target_y = PPE_CLASSES[item][2]
            for person in people:
                x1, y1, x2, y2 = person["box"]
                if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                    relative_x = (center_x - x1) / max(1, x2 - x1)
                    relative_y = (center_y - y1) / max(1, y2 - y1)
                    score = abs(relative_y - target_y) + 0.25 * abs(relative_x - 0.5)
                    candidates.append((score, person))
            if candidates:
                person = min(candidates, key=lambda pair: pair[0])[1]
                old = person["ppe_observations"].get(item)
                if old is None or detected["confidence"] > old["confidence"]:
                    person["ppe_observations"][item] = detected

        # A closed-site alarm is generated once per tracked ID per day, even
        # when the person is outside every configured PPE zone.
        local_now = datetime.now().astimezone()
        closing_time = settings.get("closing_time", "")
        site_closed = bool(closing_time and local_now.strftime("%H:%M") >= closing_time)
        after_hours_people = 0
        day_key = local_now.date().isoformat()
        self._after_hours_alerted = {
            key for key in self._after_hours_alerted if key[0] == day_key
        }
        if site_closed:
            item_names = {"hardhat": "Hardhat", "mask": "Mask", "safety vest": "Safety Vest"}
            for person in people:
                track_id = person["id"]
                if track_id is None:
                    continue
                after_hours_people += 1
                alert_key = (day_key, track_id)
                if alert_key in self._after_hours_alerted:
                    continue
                current_votes = {}
                for item, observation in person["ppe_observations"].items():
                    current_votes[item_names[item]] = {
                        observation["name"]: observation["confidence"]
                    }
                closing_summary, closing_detail = self._ppe_status(
                    current_votes, ["Hardhat", "Mask", "Safety Vest"]
                )
                self._record_event(
                    {"name": "Whole site"},
                    track_id,
                    "after-hours",
                    person["confidence"],
                    frame,
                    ppe_status=closing_detail,
                    ppe_summary=closing_summary,
                )
                self._after_hours_alerted.add(alert_key)

        for zone in zones:
            polygon = [
                (int(point["x"] * width), int(point["y"] * height))
                for point in zone["points"]
            ]
            if len(polygon) >= 3:
                cv2.polylines(frame, [np.array(polygon, dtype=np.int32)], True, (255, 190, 50), 3)
                cv2.putText(
                    frame,
                    zone["name"],
                    polygon[0],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 190, 50),
                    2,
                    cv2.LINE_AA,
                )

            for person in people:
                track_id = person["id"]
                if track_id is None:
                    continue
                if not self._person_in_zone(person["box"], polygon):
                    continue

                # Log zone presence separately from PPE violations so each
                # tracked person gets an evidence record after the dwell time.
                dwell_key = (zone["id"], track_id, "zone-dwell")
                active_now.add(dwell_key)
                dwell_state = self._active.get(dwell_key)
                if dwell_state is None or now - dwell_state["last_seen"] > ZONE_EXIT_GRACE_SECONDS:
                    dwell_state = {"since": now, "last_seen": now, "logged": False, "votes": {}}
                    self._active[dwell_key] = dwell_state
                dwell_state["last_seen"] = now

                for rule in zone["required"]:
                    # Zone configuration stores display labels (e.g. "Hardhat"),
                    # while detector associations use normalized lowercase keys.
                    observation = person["ppe_observations"].get(rule.lower())
                    if observation:
                        votes = dwell_state["votes"].setdefault(rule, {})
                        class_name = observation["name"]
                        votes[class_name] = votes.get(class_name, 0.0) + observation["confidence"]

                summary, detail = self._ppe_status(dwell_state["votes"], zone["required"])
                elapsed = now - dwell_state["since"]
                if "PPE violation" in summary:
                    ppe_alerts.add((zone["id"], track_id))
                occupants.append({
                    "zone": zone["name"],
                    "track_id": track_id,
                    "dwell_seconds": round(elapsed, 1),
                    "ppe_summary": summary,
                    "ppe_status": detail,
                    "confidence": round(float(person.get("confidence", 0.0)), 3),
                })
                if not dwell_state["logged"] and elapsed >= PERSIST_SECONDS:
                    self._record_event(
                        zone,
                        track_id,
                        "zone-dwell",
                        person.get("confidence", 0.0),
                        frame,
                        ppe_status=detail,
                        ppe_summary=summary,
                    )
                    dwell_state["logged"] = True
                color = (0, 0, 255) if "violation" in summary.lower() else (70, 225, 165)
                cv2.putText(
                    frame,
                    f"ZONE {elapsed:.1f}s | ID {track_id} | {summary}",
                    (max(8, x1), min(height - 12, max(62, y1 + 28))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        for key, state in list(self._active.items()):
            expiry = ZONE_EXIT_GRACE_SECONDS if key[2] == "zone-dwell" else LOST_TRACK_SECONDS
            if key not in active_now and now - state["last_seen"] > expiry:
                del self._active[key]

        zone_people = len({(key[0], key[1]) for key in active_now if key[2] == "zone-dwell"})
        alert_count = len(ppe_alerts)
        machinery_count = frame_counts.get("machinery", 0)
        status = f"PEOPLE: {len(people)} | IN ZONES: {zone_people} | PPE ALERTS: {alert_count} | MACHINERY: {machinery_count}"
        with self._lock:
            self._snapshot = {
                "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "tracked_people": len(people),
                "zone_people": occupants,
                "class_counts": frame_counts,
                "machinery_count": machinery_count,
                "ppe_alert_count": alert_count,
                "site_closed": site_closed,
                "closing_time": closing_time,
                "after_hours_people": after_hours_people,
            }
        cv2.rectangle(frame, (0, 0), (width, 40), (20, 27, 30), -1)
        cv2.putText(
            frame,
            status,
            (12, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (70, 225, 165) if alert_count == 0 else (50, 65, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    def stream(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open camera {camera_index}.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        try:
            while True:
                success, frame = cap.read()
                if not success or frame is None:
                    break
                annotated = self.process(frame)
                ok, buffer = cv2.imencode(".jpg", annotated)
                if ok:
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                        + buffer.tobytes()
                        + b"\r\n"
                    )
        finally:
            cap.release()


_monitor = None
_monitor_lock = threading.Lock()


def get_monitor():
    global _monitor
    with _monitor_lock:
        if _monitor is None:
            _monitor = SafetyMonitor()
        return _monitor
