from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


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


@dataclass
class SingleSideDiagnostics:
    image_rgb: np.ndarray
    top: int
    bottom: int
    left: int
    right: int
    lr_ratio: float
    tb_ratio: float
    centering_score: float
    anomaly_density: float
    anomaly_mask: np.ndarray
    whitening_ratio: float
    whitening_score: float


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

    def _surface_metrics(self, img: np.ndarray) -> Tuple[float, float, np.ndarray]:
        pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        blur = pil.filter(ImageFilter.GaussianBlur(radius=1.6))

        arr = np.asarray(pil, dtype=np.float32).mean(axis=2)
        smooth = np.asarray(blur, dtype=np.float32).mean(axis=2)

        high_freq = np.abs(arr - smooth)
        anomaly_mask = high_freq > self.anomaly_threshold
        anomaly_density = float(anomaly_mask.mean())

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

        return anomaly_density, score, anomaly_mask

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

    def analyze_side(self, image_path: Path) -> SingleSideDiagnostics:
        img = self._crop_to_card_region(self._load_image(image_path))
        gray = img.mean(axis=2)

        top, bottom, left, right = self._estimate_inner_art_bounds(gray)
        lr_ratio = min(left, right) / max(left, right)
        tb_ratio = min(top, bottom) / max(top, bottom)
        centering_score = (self._ratio_score(lr_ratio) + self._ratio_score(tb_ratio)) / 2.0

        anomaly_density, _, anomaly_mask = self._surface_metrics(img)
        whitening_ratio, whitening_score = self._edge_metrics(img)

        return SingleSideDiagnostics(
            image_rgb=np.clip(img, 0, 255).astype(np.uint8),
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            lr_ratio=float(lr_ratio),
            tb_ratio=float(tb_ratio),
            centering_score=float(centering_score),
            anomaly_density=float(anomaly_density),
            anomaly_mask=anomaly_mask,
            whitening_ratio=float(whitening_ratio),
            whitening_score=float(whitening_score),
        )

    def _build_checks(self, front_diag: SingleSideDiagnostics, back_diag: Optional[SingleSideDiagnostics]) -> CardChecks:
        lr = front_diag.lr_ratio
        tb = front_diag.tb_ratio

        centering = front_diag.centering_score
        surface_density = front_diag.anomaly_density
        surface = self._surface_score_from_density(front_diag.anomaly_density)
        whitening_ratio = front_diag.whitening_ratio
        edge = front_diag.whitening_score

        if back_diag is not None:
            centering = (centering + back_diag.centering_score) / 2
            surface = (surface + self._surface_score_from_density(back_diag.anomaly_density)) / 2
            edge = (edge + back_diag.whitening_score) / 2
            surface_density = (surface_density + back_diag.anomaly_density) / 2
            whitening_ratio = (whitening_ratio + back_diag.whitening_ratio) / 2

        overall = 0.45 * centering + 0.35 * surface + 0.20 * edge

        return CardChecks(
            centering_lr_ratio=round(lr, 4),
            centering_tb_ratio=round(tb, 4),
            centering_score=round(centering, 2),
            surface_anomaly_density=round(surface_density, 4),
            surface_score=round(surface, 2),
            edge_whitening_ratio=round(whitening_ratio, 4),
            edge_score=round(edge, 2),
            overall_score=round(overall, 2),
        )

    def _surface_score_from_density(self, anomaly_density: float) -> float:
        if anomaly_density < 0.015:
            return 10.0
        if anomaly_density < 0.025:
            return 9.0
        if anomaly_density < 0.040:
            return 8.0
        if anomaly_density < 0.060:
            return 7.0
        if anomaly_density < 0.085:
            return 6.0
        if anomaly_density < 0.110:
            return 5.0
        return 4.0

    def grade(self, front: Path, back: Optional[Path] = None) -> CardChecks:
        front_diag = self.analyze_side(front)
        back_diag = self.analyze_side(back) if back is not None else None
        return self._build_checks(front_diag, back_diag)


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


def render_diagnostic_overlay(diag: SingleSideDiagnostics, title: str) -> Image.Image:
    """Create an image preview with centering markers and surface anomaly dots."""
    base = Image.fromarray(diag.image_rgb).convert("RGB")
    draw = ImageDraw.Draw(base)

    w, h = base.size
    x_left = diag.left
    x_right = w - diag.right
    y_top = diag.top
    y_bottom = h - diag.bottom

    # Centering markers: inner-art bounds and center crosshair
    draw.rectangle([(x_left, y_top), (x_right, y_bottom)], outline=(0, 255, 255), width=3)
    draw.line([(w // 2, 0), (w // 2, h)], fill=(255, 255, 0), width=2)
    draw.line([(0, h // 2), (w, h // 2)], fill=(255, 255, 0), width=2)

    # Border margin indicators
    draw.line([(x_left, y_top), (x_left, h - 1)], fill=(255, 140, 0), width=2)
    draw.line([(x_right, y_top), (x_right, h - 1)], fill=(255, 140, 0), width=2)
    draw.line([(x_left, y_top), (w - 1, y_top)], fill=(255, 140, 0), width=2)
    draw.line([(x_left, y_bottom), (w - 1, y_bottom)], fill=(255, 140, 0), width=2)

    # Surface imperfection markers: sampled anomaly points in red
    ys, xs = np.where(diag.anomaly_mask)
    if len(xs) > 0:
        sample_step = max(1, len(xs) // 1500)
        for x, y in zip(xs[::sample_step], ys[::sample_step]):
            draw.point((int(x), int(y)), fill=(255, 0, 0))

    summary = (
        f"{title} | Center L/R {diag.lr_ratio:.3f} T/B {diag.tb_ratio:.3f} | "
        f"Surface {diag.anomaly_density:.4f} | EdgeWhite {diag.whitening_ratio:.4f}"
    )
    draw.rectangle([(5, 5), (min(w - 5, 620), 30)], fill=(0, 0, 0))
    draw.text((10, 10), summary, fill=(255, 255, 255))

    return base


def _thumbnail_for_gui(image: Image.Image, max_size: Tuple[int, int] = (360, 360)) -> Image.Image:
    preview = image.copy()
    preview.thumbnail(max_size)
    return preview


def launch_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import ImageTk

    grader = TagLikeCardGrader()

    root = tk.Tk()
    root.title("Pokémon Card Grading Check")
    root.geometry("1180x760")

    front_path = tk.StringVar()
    back_path = tk.StringVar()

    front_preview_label = ttk.Label(root)
    back_preview_label = ttk.Label(root)
    front_preview_label.image = None
    back_preview_label.image = None

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

    def set_preview(label: ttk.Label, overlay_image: Image.Image) -> None:
        thumb = _thumbnail_for_gui(overlay_image)
        tk_img = ImageTk.PhotoImage(thumb)
        label.configure(image=tk_img)
        label.image = tk_img

    def run_grading() -> None:
        if not front_path.get().strip():
            messagebox.showerror("Missing front image", "Please choose a front card image first.")
            return

        try:
            front = Path(front_path.get())
            back = Path(back_path.get()) if back_path.get().strip() else None

            front_diag = grader.analyze_side(front)
            back_diag = grader.analyze_side(back) if back is not None else None
            checks = grader._build_checks(front_diag, back_diag)

            front_overlay = render_diagnostic_overlay(front_diag, "Front")
            set_preview(front_preview_label, front_overlay)

            if back_diag is not None:
                back_overlay = render_diagnostic_overlay(back_diag, "Back")
                set_preview(back_preview_label, back_overlay)
            else:
                back_preview_label.configure(image="")
                back_preview_label.image = None

            output.configure(state="normal")
            output.delete("1.0", tk.END)
            output.insert(tk.END, format_checks(checks))
            output.configure(state="disabled")
        except Exception as exc:  # Best-effort UI error path
            messagebox.showerror("Grading failed", f"Could not grade card image(s):\n{exc}")

    controls = ttk.Frame(root, padding=10)
    controls.pack(fill="x")

    ttk.Label(controls, text="Front image:").grid(row=0, column=0, sticky="w", pady=6)
    ttk.Entry(controls, textvariable=front_path, width=85).grid(row=0, column=1, sticky="ew", pady=6)
    ttk.Button(controls, text="Select Front", command=choose_front).grid(row=0, column=2, padx=(8, 0), pady=6)

    ttk.Label(controls, text="Back image (optional):").grid(row=1, column=0, sticky="w", pady=6)
    ttk.Entry(controls, textvariable=back_path, width=85).grid(row=1, column=1, sticky="ew", pady=6)
    ttk.Button(controls, text="Select Back", command=choose_back).grid(row=1, column=2, padx=(8, 0), pady=6)

    ttk.Button(controls, text="Start Grading", command=run_grading).grid(row=2, column=1, sticky="w", pady=(8, 12))
    controls.columnconfigure(1, weight=1)

    previews = ttk.Frame(root, padding=(10, 0, 10, 10))
    previews.pack(fill="both", expand=True)

    left_panel = ttk.LabelFrame(previews, text="Front diagnostics", padding=8)
    left_panel.pack(side="left", fill="both", expand=True, padx=(0, 6))
    right_panel = ttk.LabelFrame(previews, text="Back diagnostics", padding=8)
    right_panel.pack(side="left", fill="both", expand=True, padx=(6, 0))

    front_preview_label.pack(in_=left_panel, fill="both", expand=True)
    back_preview_label.pack(in_=right_panel, fill="both", expand=True)

    output = tk.Text(root, width=110, height=10, wrap="word", state="disabled")
    output.pack(fill="x", padx=10, pady=(0, 10))

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
