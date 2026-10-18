# SafeSight
### Key tasks performed by the project

- **Detects** people, PPE items (hardhats, masks, safety vests), and machinery in images and video using the trained YOLO11 model.
- **Tracks people** across video frames so the system can follow their movement.
- **Checks zones** configured on the camera view and records when tracked people enter or remain in them.
- **Assesses PPE** for tracked people and flags detected missing PPE, with uncertain detections treated cautiously.
- **Monitors after-hours activity** using the configured site closing time and raises an alert when a person is detected after closing.
- **Displays a live dashboard** with camera feed, detection counts, zone occupancy, PPE alerts, machinery detections, and recent events.
- **Records events and evidence**, including timestamps, track IDs, event details, and saved image evidence for review.

In short, it turns camera detections into a basic site-safety monitoring and event-review workflow.
