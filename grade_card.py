from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter


@dataclass
class CardChecks:
    centering_lr_ratio: float
    centering_tb_ratio: float
    centering_score: float
    surface_anomaly_density: float
    surface_score: float
    edge_whitening_ratio: float
    edge_score: float
    overall_score: float


class TagLikeCardGrader:
    """
    Heuristic grader inspired by common grading checks.

    This is not official TAG logic; it is a configurable baseline.
    """

    def __init__(
        self,
        border_std_multiplier: float = 2.0,
        edge_strip_ratio: float = 0.04,
        anomaly_threshold: float = 20.0,
        whitening_threshold: float = 220.0,
    ) -> None:
        self.border_std_multiplier = border_std_multiplier
        self.edge_strip_ratio = edge_strip_ratio
        self.anomaly_threshold = anomaly_threshold
        self.whitening_threshold = whitening_threshold

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        return np.asarray(img, dtype=np.float32)

    @staticmethod
    def _crop_to_card_region(img: np.ndarray) -> np.ndarray:
        """Roughly isolate the card if there is a plain background."""
        gray = img.mean(axis=2)
        border_samples = np.concatenate(
            [
                gray[:10, :].ravel(),
                gray[-10:, :].ravel(),
                gray[:, :10].ravel(),
                gray[:, -10:].ravel(),
            ]
        )
        bg = np.median(border_samples)
        mask = np.abs(gray - bg) > 8.0

        ys, xs = np.where(mask)
        if len(xs) < 0.2 * gray.size:
            return img

        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()

        if (y1 - y0) < img.shape[0] * 0.5 or (x1 - x0) < img.shape[1] * 0.5:
            return img

        return img[y0 : y1 + 1, x0 : x1 + 1]

    def _estimate_inner_art_bounds(self, gray: np.ndarray) -> Tuple[int, int, int, int]:
        """Estimate where the high-detail artwork starts by scanning inward."""
        h, w = gray.shape
        row_std = gray.std(axis=1)
        col_std = gray.std(axis=0)

        target_row = np.median(row_std) * self.border_std_multiplier
        target_col = np.median(col_std) * self.border_std_multiplier

        max_scan_h = max(5, int(h * 0.25))
        max_scan_w = max(5, int(w * 0.25))

        top = next((i for i in range(max_scan_h) if row_std[i] > target_row), int(h * 0.08))
        bottom = next(
            (i for i in range(max_scan_h) if row_std[h - 1 - i] > target_row),
            int(h * 0.08),
        )
        left = next((i for i in range(max_scan_w) if col_std[i] > target_col), int(w * 0.08))
        right = next(
            (i for i in range(max_scan_w) if col_std[w - 1 - i] > target_col),
            int(w * 0.08),
        )

        top = int(np.clip(top, 1, h // 3))
        bottom = int(np.clip(bottom, 1, h // 3))
        left = int(np.clip(left, 1, w // 3))
        right = int(np.clip(right, 1, w // 3))
        return top, bottom, left, right

    @staticmethod
    def _ratio_score(balance_ratio: float) -> float:
        """Convert centering balance into an approximate 1-10 score."""
        ratio = float(np.clip(balance_ratio, 0.0, 1.0))
        if ratio >= 0.95:
            return 10.0
        if ratio >= 0.90:
            return 9.5
        if ratio >= 0.85:
            return 9.0
        if ratio >= 0.80:
            return 8.0
        if ratio >= 0.75:
            return 7.0
        if ratio >= 0.70:
            return 6.0
        if ratio >= 0.65:
            return 5.0
        if ratio >= 0.60:
            return 4.0
        return 3.0

    def _centering_metrics(self, img: np.ndarray) -> Tuple[float, float, float]:
        gray = img.mean(axis=2)
        top, bottom, left, right = self._estimate_inner_art_bounds(gray)

        lr_ratio = min(left, right) / max(left, right)
        tb_ratio = min(top, bottom) / max(top, bottom)
        center_score = (self._ratio_score(lr_ratio) + self._ratio_score(tb_ratio)) / 2.0
        return float(lr_ratio), float(tb_ratio), float(center_score)

    def _surface_metrics(self, img: np.ndarray) -> Tuple[float, float]:
        pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        blur = pil.filter(ImageFilter.GaussianBlur(radius=1.6))

        arr = np.asarray(pil, dtype=np.float32).mean(axis=2)
        smooth = np.asarray(blur, dtype=np.float32).mean(axis=2)

        high_freq = np.abs(arr - smooth)
        anomaly_density = float((high_freq > self.anomaly_threshold).mean())

        if anomaly_density < 0.015:
            score = 10.0
        elif anomaly_density < 0.025:
            score = 9.0
        elif anomaly_density < 0.040:
            score = 8.0
        elif anomaly_density < 0.060:
            score = 7.0
        elif anomaly_density < 0.085:
            score = 6.0
        elif anomaly_density < 0.110:
            score = 5.0
        else:
            score = 4.0

        return anomaly_density, score

    def _edge_metrics(self, img: np.ndarray) -> Tuple[float, float]:
        gray = img.mean(axis=2)
        h, w = gray.shape
        t = max(1, int(min(h, w) * self.edge_strip_ratio))

        edges = np.concatenate(
            [
                gray[:t, :].ravel(),
                gray[-t:, :].ravel(),
                gray[:, :t].ravel(),
                gray[:, -t:].ravel(),
            ]
        )

        whitening_ratio = float((edges > self.whitening_threshold).mean())

        if whitening_ratio < 0.01:
            score = 10.0
        elif whitening_ratio < 0.02:
            score = 9.0
        elif whitening_ratio < 0.035:
            score = 8.0
        elif whitening_ratio < 0.06:
            score = 7.0
        elif whitening_ratio < 0.09:
            score = 6.0
        elif whitening_ratio < 0.13:
            score = 5.0
        else:
            score = 4.0

        return whitening_ratio, score

    def grade(self, front: Path, back: Optional[Path] = None) -> CardChecks:
        front_img = self._crop_to_card_region(self._load_image(front))

        lr, tb, centering = self._centering_metrics(front_img)
        surface_density, surface = self._surface_metrics(front_img)
        whitening_ratio, edge = self._edge_metrics(front_img)

        if back is not None:
            back_img = self._crop_to_card_region(self._load_image(back))
            _, _, back_center = self._centering_metrics(back_img)
            back_surface_density, back_surface = self._surface_metrics(back_img)
            back_whitening_ratio, back_edge = self._edge_metrics(back_img)

            centering = (centering + back_center) / 2
            surface = (surface + back_surface) / 2
            edge = (edge + back_edge) / 2
            surface_density = (surface_density + back_surface_density) / 2
            whitening_ratio = (whitening_ratio + back_whitening_ratio) / 2

        overall = 0.45 * centering + 0.35 * surface + 0.20 * edge

        return CardChecks(
            centering_lr_ratio=lr,
            centering_tb_ratio=tb,
            centering_score=round(centering, 2),
            surface_anomaly_density=round(surface_density, 4),
            surface_score=round(surface, 2),
            edge_whitening_ratio=round(whitening_ratio, 4),
            edge_score=round(edge, 2),
            overall_score=round(overall, 2),
        )


def format_checks(checks: CardChecks) -> str:
    return "\n".join(
        [
            "=== Pokémon Card Check (TAG-style baseline) ===",
            f"Centering L/R ratio : {checks.centering_lr_ratio:.3f}",
            f"Centering T/B ratio : {checks.centering_tb_ratio:.3f}",
            f"Centering score     : {checks.centering_score:.2f}/10",
            f"Surface anomaly     : {checks.surface_anomaly_density:.4f}",
            f"Surface score       : {checks.surface_score:.2f}/10",
            f"Edge whitening      : {checks.edge_whitening_ratio:.4f}",
            f"Edge score          : {checks.edge_score:.2f}/10",
            f"Overall score       : {checks.overall_score:.2f}/10",
        ]
    )


def launch_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    grader = TagLikeCardGrader()

    root = tk.Tk()
    root.title("Pokémon Card Grading Check")
    root.geometry("740x420")

    front_path = tk.StringVar()
    back_path = tk.StringVar()

    def choose_front() -> None:
        picked = filedialog.askopenfilename(
            title="Select front card image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff")],
        )
        if picked:
            front_path.set(picked)

    def choose_back() -> None:
        picked = filedialog.askopenfilename(
            title="Select back card image (optional)",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff")],
        )
        if picked:
            back_path.set(picked)

    def run_grading() -> None:
        if not front_path.get().strip():
            messagebox.showerror("Missing front image", "Please choose a front card image first.")
            return

        try:
            front = Path(front_path.get())
            back = Path(back_path.get()) if back_path.get().strip() else None
            checks = grader.grade(front=front, back=back)
            output.configure(state="normal")
            output.delete("1.0", tk.END)
            output.insert(tk.END, format_checks(checks))
            output.configure(state="disabled")
        except Exception as exc:  # Best-effort UI error path
            messagebox.showerror("Grading failed", f"Could not grade card image(s):\n{exc}")

    frm = ttk.Frame(root, padding=12)
    frm.pack(fill="both", expand=True)

    ttk.Label(frm, text="Front image:").grid(row=0, column=0, sticky="w", pady=6)
    ttk.Entry(frm, textvariable=front_path, width=70).grid(row=0, column=1, sticky="ew", pady=6)
    ttk.Button(frm, text="Select Front", command=choose_front).grid(row=0, column=2, padx=(8, 0), pady=6)

    ttk.Label(frm, text="Back image (optional):").grid(row=1, column=0, sticky="w", pady=6)
    ttk.Entry(frm, textvariable=back_path, width=70).grid(row=1, column=1, sticky="ew", pady=6)
    ttk.Button(frm, text="Select Back", command=choose_back).grid(row=1, column=2, padx=(8, 0), pady=6)

    ttk.Button(frm, text="Start Grading", command=run_grading).grid(row=2, column=1, sticky="w", pady=(10, 10))

    output = tk.Text(frm, width=90, height=15, wrap="word", state="disabled")
    output.grid(row=3, column=0, columnspan=3, sticky="nsew")

    frm.columnconfigure(1, weight=1)
    frm.rowconfigure(3, weight=1)

    root.mainloop()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TAG-style Pokémon card grading baseline")
    parser.add_argument("--front", type=Path, help="Path to front card image")
    parser.add_argument("--back", type=Path, default=None, help="Path to back card image")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    parser.add_argument("--gui", action="store_true", help="Launch the GUI card selector")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.gui:
        launch_gui()
        return

    if args.front is None:
        raise SystemExit("error: --front is required unless --gui is used")

    grader = TagLikeCardGrader()
    checks = grader.grade(front=args.front, back=args.back)

    if args.json:
        print(json.dumps(asdict(checks), indent=2))
        return

    print(format_checks(checks))


if __name__ == "__main__":
    main()
