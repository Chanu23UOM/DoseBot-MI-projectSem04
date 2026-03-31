"""
DoseBot — Computer Vision Pipeline
====================================
Monitors the dispensing chute via webcam, classifies each falling
candy's colour using HSV thresholding, and broadcasts results to
LabVIEW over a local UDP socket.

Usage:
    python dosebot_cv_pipeline.py --target RED --count 5

Dependencies:
    pip install opencv-python numpy

LabVIEW side:
    Use a UDP Read node on port 5005 to receive JSON strings:
    e.g.  {"color": "RED", "match": true, "pill_count": 3, "status": "DISPENSING"}
"""

import cv2
import numpy as np
import socket
import json
import argparse
import time
import sys

# ── UDP Configuration ─────────────────────────────────────────────────────
LABVIEW_IP   = "127.0.0.1"   # Localhost — both scripts on same machine
LABVIEW_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ── HSV Colour Ranges for M&M / Skittles Simulation ─────────────────────
# Tune these ranges under your actual lab lighting conditions.
# Use the calibration mode (--calibrate flag) to find correct values.
COLOUR_RANGES = {
    "RED": [
        (np.array([0,   120,  70]),  np.array([10,  255, 255])),   # Red wraps in HSV
        (np.array([170, 120,  70]),  np.array([180, 255, 255])),   # Red upper wrap
    ],
    "GREEN": [
        (np.array([36,  80,   40]),  np.array([86,  255, 255])),
    ],
    "BLUE": [
        (np.array([100, 80,   40]),  np.array([130, 255, 255])),
    ],
    "YELLOW": [
        (np.array([22,  80,   40]),  np.array([35,  255, 255])),
    ],
    "ORANGE": [
        (np.array([11,  120,  70]),  np.array([21,  255, 255])),
    ],
}

# ── Detection Parameters ──────────────────────────────────────────────────
MIN_CONTOUR_AREA   = 300    # px² — ignore dust/noise below this size
DETECTION_ROI_Y1   = 0.35   # Region of Interest: top 35% of frame (chute zone)
DETECTION_ROI_Y2   = 0.75   # to bottom 75%
COOLDOWN_FRAMES    = 18     # Frames to ignore after a detection (prevents double-count)
MOTION_THRESHOLD   = 800    # Min frame-diff pixel count to consider "something is moving"


def send_to_labview(payload: dict):
    """Serialise payload to JSON and UDP-broadcast to LabVIEW."""
    msg = json.dumps(payload).encode("utf-8")
    sock.sendto(msg, (LABVIEW_IP, LABVIEW_PORT))
    print(f"  → LabVIEW: {payload}")


def build_colour_mask(hsv_frame, colour_name: str) -> np.ndarray:
    """Return a binary mask for the given named colour."""
    ranges = COLOUR_RANGES.get(colour_name.upper(), [])
    mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
    for (lo, hi) in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv_frame, lo, hi))
    # Morphological clean-up: close small holes, remove isolated noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, mask=None, kernel=kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  mask=None, kernel=kernel)
    return mask


def classify_dominant_colour(hsv_frame) -> str:
    """Return the colour name with the largest detected blob, or 'UNKNOWN'."""
    best_colour = "UNKNOWN"
    best_area   = 0
    for colour_name in COLOUR_RANGES:
        mask = build_colour_mask(hsv_frame, colour_name)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > MIN_CONTOUR_AREA and area > best_area:
                best_area   = area
                best_colour = colour_name
    return best_colour


def run_calibration_mode(cap):
    """
    Calibration helper: displays live HSV values at the mouse cursor
    so you can tune COLOUR_RANGES for your specific lighting setup.
    Press Q to exit.
    """
    print("\n[CALIBRATION MODE] Hover your cursor over a candy to read its HSV values.")
    print("Update COLOUR_RANGES at the top of this file with the values you observe.")
    print("Press Q to exit calibration.\n")

    hsv_vals = [0, 0, 0]

    def mouse_callback(event, x, y, flags, param):
        nonlocal hsv_vals
        if event == cv2.EVENT_MOUSEMOVE:
            hsv_vals = param[y, x].tolist()

    cv2.namedWindow("DoseBot Calibration")
    cv2.setMouseCallback("DoseBot Calibration", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.setMouseCallback("DoseBot Calibration", mouse_callback, hsv)

        h, s, v = hsv_vals
        overlay = frame.copy()
        cv2.putText(overlay, f"HSV: H={h}  S={s}  V={v}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imshow("DoseBot Calibration", overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def run_detection(target_colour: str, target_count: int, camera_index: int = 0, debug: bool = False):
    """
    Main detection loop.

    Args:
        target_colour:  Colour string matching a key in COLOUR_RANGES (e.g. "RED")
        target_count:   Number of matching pills the prescription requires
        camera_index:   OpenCV camera index (default 0 = first webcam)
        debug:          Show annotated live video window
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {camera_index}. Check USB connection.")
        sys.exit(1)

    # Force a reasonable resolution for reliable detection
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"\n[DoseBot CV] Target: {target_count}x {target_colour} pills")
    print(f"[DoseBot CV] Broadcasting detections → {LABVIEW_IP}:{LABVIEW_PORT}\n")

    pill_count       = 0
    cooldown_counter = 0
    prev_gray        = None

    # Notify LabVIEW system is ready
    send_to_labview({"color": "NONE", "match": False, "pill_count": 0, "status": "READY"})

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame — retrying...")
            time.sleep(0.05)
            continue

        h, w = frame.shape[:2]

        # ── Define ROI (chute zone only) ─────────────────────────────────
        roi_y1 = int(h * DETECTION_ROI_Y1)
        roi_y2 = int(h * DETECTION_ROI_Y2)
        roi    = frame[roi_y1:roi_y2, :]

        # ── Motion detection: only classify when something is moving ─────
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)

        motion_detected = False
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion_pixels = np.sum(diff > 25)
            motion_detected = motion_pixels > MOTION_THRESHOLD
        prev_gray = gray

        detected_colour = "NONE"
        is_match        = False

        if motion_detected and cooldown_counter == 0:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            detected_colour = classify_dominant_colour(hsv)
            is_match = (detected_colour == target_colour.upper())

            if detected_colour != "UNKNOWN" and detected_colour != "NONE":
                if is_match:
                    pill_count += 1
                    status = "DISPENSING" if pill_count < target_count else "COMPLETE"
                    print(f"[✓] {detected_colour} — MATCH   | Count: {pill_count}/{target_count}")
                else:
                    status = "ROGUE_PILL"
                    print(f"[✗] {detected_colour} — MISMATCH (expected {target_colour})")

                send_to_labview({
                    "color":      detected_colour,
                    "match":      is_match,
                    "pill_count": pill_count,
                    "status":     status
                })

                cooldown_counter = COOLDOWN_FRAMES  # Prevent double-counting

                # If prescription fulfilled, signal completion
                if pill_count >= target_count:
                    print(f"\n[DoseBot CV] Prescription complete: {pill_count}x {target_colour}")
                    time.sleep(0.5)
                    send_to_labview({
                        "color":      target_colour,
                        "match":      True,
                        "pill_count": pill_count,
                        "status":     "COMPLETE"
                    })
                    break

        if cooldown_counter > 0:
            cooldown_counter -= 1

        # ── Debug visualisation ─────────────────────────────────────────
        if debug:
            display = frame.copy()
            # Draw ROI boundary
            cv2.rectangle(display, (0, roi_y1), (w, roi_y2), (0, 255, 0), 2)
            # Overlay stats
            cv2.putText(display, f"Target:  {target_count}x {target_colour}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, f"Counted: {pill_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
            cv2.putText(display, f"Last:    {detected_colour}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if is_match else (0, 0, 255), 2)
            if motion_detected:
                cv2.putText(display, "MOTION", (w - 140, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            cv2.imshow("DoseBot CV — Press Q to abort", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                send_to_labview({"color": "NONE", "match": False,
                                 "pill_count": pill_count, "status": "ABORTED"})
                break

    cap.release()
    cv2.destroyAllWindows()
    sock.close()
    print("[DoseBot CV] Shutdown complete.")


# ── Entry Point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DoseBot CV Pipeline")
    parser.add_argument("--target",    type=str, default="RED",
                        help="Target pill colour (RED, GREEN, BLUE, YELLOW, ORANGE)")
    parser.add_argument("--count",     type=int, default=5,
                        help="Number of target-colour pills in prescription")
    parser.add_argument("--camera",    type=int, default=0,
                        help="OpenCV camera index (default: 0)")
    parser.add_argument("--debug",     action="store_true",
                        help="Show annotated live video window")
    parser.add_argument("--calibrate", action="store_true",
                        help="Enter calibration mode to tune HSV colour ranges")
    args = parser.parse_args()

    if args.calibrate:
        cap = cv2.VideoCapture(args.camera)
        run_calibration_mode(cap)
        cap.release()
    else:
        if args.target.upper() not in COLOUR_RANGES:
            print(f"[ERROR] Unknown colour '{args.target}'. "
                  f"Valid options: {list(COLOUR_RANGES.keys())}")
            sys.exit(1)
        run_detection(
            target_colour=args.target,
            target_count=args.count,
            camera_index=args.camera,
            debug=args.debug
        )
