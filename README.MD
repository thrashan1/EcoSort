# EcoSort AI

A simple AI tool that helps people sort waste correctly using a webcam.

## Why?

A lot of recyclable and compostable waste ends up in landfills because people either aren't sure where it goes or don't want to sort it.

Some examples:
- Food scraps get thrown in garbage instead of compost.
- Plastic bottles and cans end up in landfill.
- Recycling bins become contaminated because the wrong items are thrown in.

Our goal is to make sorting waste as easy as showing it to a camera.

---

## How it Works

1. Open the application.
2. Hold an item in front of the webcam.
3. The AI predicts whether it's:
   - ♻️ Recycling
   - 🌱 Compost
   - 🗑️ Garbage
4. The result is shown on screen with a confidence score.

The idea is that this could eventually be built into public recycling stations, apartment garbage rooms, schools, or even smart garbage bins.

---

## Current Progress

- [x] Project idea finalized
- [x] Tech stack decided
- [x] Repository created
- [ ] Webcam integration
- [ ] Dataset collection
- [ ] Model training
- [ ] Live classification
- [ ] UI improvements

---

## Tech Stack

- Python
- OpenCV
- TensorFlow / Keras
- MobileNetV2
- Git & GitHub

---

## Project Structure

```
EcoSort-AI/
│
├── dataset/
│   ├── recycling/
│   ├── compost/
│   └── garbage/
│
├── models/
│
├── src/
│   ├── webcam.py
│   ├── train.py
│   ├── predict.py
│   └── ui.py
│
├── requirements.txt
└── README.md
```

---

## Future Ideas

- Detect more waste categories (glass, metal, paper, electronics)
- Add voice feedback
- Automatically open the correct bin using a servo motor
- Keep track of how much waste has been sorted
- Display environmental statistics, like landfill waste diverted

---

## Team

GenZ Can Hack 2026
