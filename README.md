# CPAR-Senior-Project
# Self-arranging Cupboard

This is the code for my Computing and the Arts senior project at Yale. 

Instead of building robots that have to figure out our messy homes, what if we designed homes that robots could actually read? The cupboard is my attempt at that. You load in your kitchen , press a button, and it looks at each item, figures out what it is, and shuffles the storage modules around (like a sliding puzzle) until everything is where it belongs.

Currently, this repo contains the object detection aspect of the project: the camera and computer vision that tells the cupboard what's in each box– its color, type, and size.

## What's in here

```
CPAR-Senior-Project/
├── detect.py        # the main vision script (explained below)
├── yoloe_test.py    # quick experiments with YOLOE text prompts
├── *.jpeg           # test photos of cups and mugs in the box
└── runs/            # YOLO saves its annotated output images here (not on GitHub)
```

Model files (`*.pt`, `*.ts`), the `runs/` folder, and the virtual environment are all in `.gitignore`, so they stay on my laptop. They download automatically the first time you run anything.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install ultralytics
```

To try it on a photo:

```
python detect.py
```

## detect.py

Input a photo of one item sitting in a box, and it tells you three things: **what** the item is, **what color** it is, and **how big** it is. Here's what happens step by step.

### 1. Item type detection

It runs **YOLO26n-seg**, a small, fast object detection model from Ultralytics. "Seg" means segmentation, so it traces the actual outline instead of just drawing a rectangle around the item. That outline is the key to everything else.

A few tweaks to adapt it for my project:

- **Kitchen categories only.** The YOLO model knows 80 everyday things, but I only let it look for bottle, wine glass, cup, fork, knife, spoon, bowl, and vase. 
- **Low confidence threshold (5%).** By default YOLO only reports things it's at least 25% sure about, but I adjusted it to 5% to maximize find rate.
- **Ignore the box.** If a detection covers more than 80% of the photo, that means it's detecting the box itself, not an item.
- **Keep only the best guess.** Each photo should have one item, so only the highest-confidence detection gets pushed.

### 2. Color Detection

Using the outline as a stencil, we take only the pixels that belong to the item and finds their main color. It works in **HSV** (hue, saturation, brightness) instead of RGB because categorization by hue is much easier. The `color_name()` function maps hue numbers into color words.

Detecting **clear and glass items** were the tricky part. A clear cup's "color" is the background behind it, so the script samples the background right around the item and compares the two. If the inside and outside look almost the same, the item is see-through and gets labeled **clear**. The comparison uses Lab color, which measures difference closer to how our eyes see it.

### 3. Size Measurement

It takes the box YOLO drew around the object for object detection and takes its **relative size to the photo**:

- `width`: fraction of the frame's width (0.31 = 31%)
- `height`: fraction of the frame's height
- `area`: width × height

I went with relative size instead of centimeters because what the cupboard really needs to know is "will this fit in a small, large, or narrow module," and that doesn't require calibrating a camera.

### 4. Results

The call returns a summary of information about the detected object:

```python
{
    "label": "cup",
    "confidence": 0.62,
    "color": "clear",
    "width": 0.312,
    "height": 0.58,
    "area": 0.181,
    "color_difference": 9.4,
}
```

If nothing is found, it returns `None`. The idea is that the sorting and path-solving code can just use this results directly in its algorithm.

## Tuning parameters

- **`CLEAR_THRESHOLD`** how similar inside and outside must be to count as clear. Run a few clear and solid items, look at their `color_difference` numbers, and set this in between.
- **The cutoffs in `color_name()`**: the script prints the raw numbers, so you can adjust the ranges to match the true colors.
- **`KITCHEN`**: which of the 80 built-in categories the model is allowed to look for.