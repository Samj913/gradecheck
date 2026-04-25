# Pokémon Card Grading Check AI (TAG-style baseline)

This project provides a practical **computer-vision baseline** for checking card quality with metrics similar to what collectors care about in grading:

- **Centering** (left/right and top/bottom symmetry)
- **Surface condition** (scratches / texture anomalies)
- **Edge whitening** (white chips around the border)

> ⚠️ This is **not** an official TAG Grading implementation. It is an educational approximation you can tune with your own card scan dataset.

## Quick start (CLI)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python grade_card.py --front examples/sample_front.jpg --back examples/sample_back.jpg --json
```

## GUI mode (file picker + grading button)

If you prefer a desktop window:

```bash
python grade_card.py --gui
```

The GUI provides:
- a **Select Front** button to choose the front image
- a **Select Back** button for an optional back image
- a **Start Grading** button to run grading and display the results
- a **diagnostic view** of front/back cards with centering markers and red surface-imperfection points
- **scroll + zoom controls** on previews (mouse wheel to zoom, drag to pan, scrollbars to navigate)

## Input recommendations

For best results:
- Use flatbed scans or very stable lighting.
- Keep card approximately upright and mostly fills the frame.
- Avoid heavy shadows and glare.

## How scoring works

The baseline computes these subscores (1–10):

- `centering_score`
- `surface_score`
- `edge_score`

And combines them into a weighted final score:

```text
overall = 0.45 * centering + 0.35 * surface + 0.20 * edge
```

Weights and thresholds can be tuned in `grade_card.py`.

## CLI usage

```bash
python grade_card.py --front path/to/front.jpg [--back path/to/back.jpg] [--json]
```

- `--json`: print machine-readable output.
- `--gui`: open the graphical interface instead of CLI input.

## Limitations

- This is a heuristic model, not a trained deep-learning grader.
- Real grading involves far more nuanced defect detection.
- You should calibrate thresholds using known graded cards.
