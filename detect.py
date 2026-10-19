import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")

# Kitchen categories only: bottle, wine glass, cup, fork, knife, spoon, bowl, vase
KITCHEN = [39, 40, 41, 42, 43, 44, 45, 75]

# How similar inside and outside must be to count as clear.
# Lower = stricter. Tune this by looking at the "difference" number it reports.
CLEAR_THRESHOLD = 18


def color_name(h, s, v):
    # OpenCV hue runs 0-179 (half of the usual 0-360 color wheel)
    if v < 50:
        return "black"
    if s < 40:
        return "white" if v > 180 else "grey"
    if h < 25 and v < 150:
        return "brown"
    if h < 8 or h >= 165:
        return "red"
    if h < 22:
        return "orange"
    if h < 35:
        return "yellow"
    if h < 85:
        return "green"
    if h < 130:
        return "blue"
    return "purple"


def detect(image_path):
    result = model.predict(image_path, conf=0.05, classes=KITCHEN, save=True)[0]
    img = result.orig_img

    # Find the best detection, skipping anything the size of the whole box
    best, best_score = None, 0
    for i, box in enumerate(result.boxes):
        w, h = box.xywhn[0][2], box.xywhn[0][3]
        if w * h > 0.8:
            continue
        if float(box.conf) > best_score:
            best, best_score = i, float(box.conf)

    if best is None:
        return None

    label = model.names[int(result.boxes.cls[best])]

    # Stencil of the item, the same size as the photo
    outline = result.masks.xy[best].astype(np.int32)
    stencil = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(stencil, [outline], 255)

    # Shrink the stencil a little so edge pixels don't blur the color
    size = max(5, int(img.shape[1] * 0.02))
    kernel = np.ones((size, size), np.uint8)
    inside = cv2.erode(stencil, kernel)

    # A thin ring of background just outside the item
    grown = cv2.dilate(stencil, kernel, iterations=2)
    ring = cv2.subtract(grown, stencil)

    # Compare inside vs ring in Lab color to spot clear items
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    inside_lab = np.median(lab[inside == 255], axis=0)
    ring_lab = np.median(lab[ring == 255], axis=0)
    difference = float(np.linalg.norm(inside_lab - ring_lab))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = np.median(hsv[inside == 255], axis=0)
    color = "clear" if difference < CLEAR_THRESHOLD else color_name(h, s, v)

    # Size relative to the photo: 1.0 would mean the full width/height of the frame
    rel_width = float(result.boxes.xywhn[best][2])
    rel_height = float(result.boxes.xywhn[best][3])
    rel_area = rel_width * rel_height

    return {
        "label": label,
        "confidence": round(best_score, 2),
        "color": color,
        "width": round(rel_width, 3),
        "height": round(rel_height, 3),
        "area": round(rel_area, 3),
        "color_difference": round(difference, 1),
    }


if __name__ == "__main__":
    item = detect("glass-cup.jpeg")
    if item is None:
        print("Nothing found")
    else:
        print(f"Found: {item['label']} ({item['confidence']:.0%} sure)")
        print(f"Color: {item['color']}  (difference from background {item['color_difference']})")
        print(f"Width:  {item['width']:.1%} of frame")
        print(f"Height: {item['height']:.1%} of frame")
        print(f"Area:   {item['area']:.1%} of frame")