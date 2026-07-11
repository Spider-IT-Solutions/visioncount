import json
import os

import cv2
import numpy as np

from tracker import Tracker

VIDEO_SOURCE = 'box-passing.mp4'   # swap to 0 (or 1) for a live webcam
SETTINGS_FILE = 'panel_settings.json'
FRAME_W, FRAME_H = 640, 480

DEFAULTS = {
    "l_h": 0, "l_s": 62, "l_v": 100,
    "u_h": 179, "u_s": 255, "u_v": 255,
    "roi_x": 370, "roi_y": 237, "roi_w": 40, "roi_h": 206,
    "min_area": 500,
}


def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE) as f:
            saved = json.load(f)
        settings = DEFAULTS.copy()
        settings.update(saved)
        return settings
    return DEFAULTS.copy()


def save_settings(settings):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)
    print(f"Saved calibration -> {SETTINGS_FILE}")
    print(json.dumps(settings, indent=2))


def nothing(x):
    pass


def build_controls(settings):
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 400, 400)
    cv2.createTrackbar("L-H", "Controls", settings["l_h"], 179, nothing)
    cv2.createTrackbar("L-S", "Controls", settings["l_s"], 255, nothing)
    cv2.createTrackbar("L-V", "Controls", settings["l_v"], 255, nothing)
    cv2.createTrackbar("U-H", "Controls", settings["u_h"], 179, nothing)
    cv2.createTrackbar("U-S", "Controls", settings["u_s"], 255, nothing)
    cv2.createTrackbar("U-V", "Controls", settings["u_v"], 255, nothing)
    cv2.createTrackbar("ROI X", "Controls", settings["roi_x"], FRAME_W, nothing)
    cv2.createTrackbar("ROI Y", "Controls", settings["roi_y"], FRAME_H, nothing)
    cv2.createTrackbar("ROI W", "Controls", settings["roi_w"], FRAME_W, nothing)
    cv2.createTrackbar("ROI H", "Controls", settings["roi_h"], FRAME_H, nothing)
    cv2.createTrackbar("Min Area", "Controls", settings["min_area"], 5000, nothing)


def read_controls():
    return {
        "l_h": cv2.getTrackbarPos("L-H", "Controls"),
        "l_s": cv2.getTrackbarPos("L-S", "Controls"),
        "l_v": cv2.getTrackbarPos("L-V", "Controls"),
        "u_h": cv2.getTrackbarPos("U-H", "Controls"),
        "u_s": cv2.getTrackbarPos("U-S", "Controls"),
        "u_v": cv2.getTrackbarPos("U-V", "Controls"),
        "roi_x": cv2.getTrackbarPos("ROI X", "Controls"),
        "roi_y": cv2.getTrackbarPos("ROI Y", "Controls"),
        "roi_w": max(1, cv2.getTrackbarPos("ROI W", "Controls")),
        "roi_h": max(1, cv2.getTrackbarPos("ROI H", "Controls")),
        "min_area": cv2.getTrackbarPos("Min Area", "Controls"),
    }


def main():
    settings = load_settings()
    build_controls(settings)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    is_file_source = isinstance(VIDEO_SOURCE, str)

    tracker = Tracker()
    counted_ids = set()

    print("Controls: q=quit  s=save calibration  r=reset count")

    while True:
        ret, frame = cap.read()
        if not ret:
            if is_file_source:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        s = read_controls()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([s["l_h"], s["l_s"], s["l_v"]])
        upper = np.array([s["u_h"], s["u_s"], s["u_v"]])
        mask = cv2.inRange(hsv, lower, upper)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        boxes = []
        for c in cnts:
            if cv2.contourArea(c) > s["min_area"]:
                boxes.append(list(cv2.boundingRect(c)))

        rx, ry, rw, rh = s["roi_x"], s["roi_y"], s["roi_w"], s["roi_h"]

        bbox_ids = tracker.update(boxes)
        for x1, y1, w1, h1, obj_id in bbox_ids:
            cx = x1 + w1 // 2
            cy = y1 + h1 // 2
            inside = rx <= cx <= rx + rw and ry <= cy <= ry + rh
            color = (0, 255, 0) if inside else (255, 150, 0)
            cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), color, 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, str(obj_id), (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if inside and obj_id not in counted_ids:
                counted_ids.add(obj_id)

        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (255, 255, 255), 2)
        cv2.putText(frame, f"Count: {len(counted_ids)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Live", frame)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_settings(s)
        elif key == ord('r'):
            counted_ids.clear()
            print("Count reset")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Final count this session: {len(counted_ids)}")


if __name__ == '__main__':
    main()
