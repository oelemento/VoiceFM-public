#!/usr/bin/env python3
"""Assemble all paper figures into a single PDF (vector when available).

Each manuscript figure becomes one page (or two if panels are separate).
reportlab places a raster PNG, then a pikepdf post-pass stamps the matching
vector PDF on top of every panel that has a `.pdf` companion. Result: vector
panels in the assembled output, with raster fallback for AI-generated cartoons.

Usage:
    python scripts/paper_assemble_pdf_v3.py
"""

import csv
import re
from pathlib import Path

import pikepdf
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle


_SUPER_MAP = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5",
    "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "-", "⁺": "+",
}
_SUPER_RE = re.compile("([" + "".join(_SUPER_MAP.keys()) + "]+)")


def _caption_to_html(text: str) -> str:
    """Convert Unicode superscript runs into reportlab <super>...</super> tags
    so Paragraph renders them properly (Helvetica lacks the superscript glyphs)."""
    def repl(m):
        return "<super>" + "".join(_SUPER_MAP[c] for c in m.group(1)) + "</super>"
    return _SUPER_RE.sub(repl, text).replace("&", "&amp;")


def _draw_caption(c, text, x, y_top, width, font_size=8, leading=11, y_bot=None):
    """Render a caption with auto-wrapping and inline <super> tags.

    Returns the new y after drawing. If the caption would overflow below
    y_bot, it is truncated to what fits.
    """
    style = ParagraphStyle(
        name="Caption",
        fontName="Helvetica",
        fontSize=font_size,
        leading=leading,
        alignment=0,  # left
    )
    p = Paragraph(_caption_to_html(text), style)
    avail_h = (y_top - y_bot) if y_bot is not None else y_top
    w, h = p.wrap(width, avail_h)
    p.drawOn(c, x, y_top - h)
    return y_top - h

PROJECT_ROOT = Path(__file__).parent.parent
PAPER_DIR = PROJECT_ROOT / "paper_v3"

# Page setup
PAGE_W, PAGE_H = letter
MARGIN = 0.6 * inch
USABLE_W = PAGE_W - 2 * MARGIN
TITLE_SIZE = 14
SUBTITLE_SIZE = 10

# Tracks (page_number, pdf_path, x, y_bot, w, h) for vector overlay pass.
PDF_OVERLAYS: list[tuple[int, Path, float, float, float, float]] = []


def draw_image(c, img_path, x, y, max_w, max_h=None):
    """Draw image preserving aspect ratio, fitting within max_w × max_h.

    If a `.pdf` companion exists alongside the PNG, record the panel region
    so the post-pass can stamp the vector version on top.
    """
    if not img_path.exists():
        # Draw placeholder
        c.setFont("Helvetica-Oblique", 10)
        c.setFillColorRGB(0.6, 0.6, 0.6)
        c.drawString(x, y - 20, f"[Pending: {img_path.name}]")
        c.setFillColorRGB(0, 0, 0)
        return 30
    img = Image.open(img_path)
    iw, ih = img.size
    aspect = ih / iw
    draw_w = max_w
    draw_h = draw_w * aspect
    if max_h and draw_h > max_h:
        draw_h = max_h
        draw_w = draw_h / aspect
    # reportlab y is bottom of image
    c.drawImage(str(img_path), x, y - draw_h, draw_w, draw_h,
                preserveAspectRatio=True, anchor="nw")
    pdf_path = img_path.with_suffix(".pdf")
    if pdf_path.exists():
        PDF_OVERLAYS.append((c.getPageNumber(), pdf_path, x, y - draw_h, draw_w, draw_h))
    return draw_h


def stamp_vector_overlays(pdf_path: Path) -> None:
    """Open the assembled PDF and stamp each tracked vector figure on top.

    Each overlay is added as a Form XObject scaled to the recorded panel rect,
    preserving the panel's aspect ratio. The raster PNG already drawn underneath
    is harmless — it sits behind the vector and is not visible at any zoom.
    """
    if not PDF_OVERLAYS:
        return
    pdf = pikepdf.Pdf.open(str(pdf_path), allow_overwriting_input=True)
    counter = 0
    for page_no, fig_pdf_path, x, y_bot, w, h in PDF_OVERLAYS:
        target_page = pdf.pages[page_no - 1]  # 1-indexed → 0-indexed
        with pikepdf.Pdf.open(str(fig_pdf_path)) as fig_pdf:
            fig_page = fig_pdf.pages[0]
            box = fig_page.mediabox
            fw = float(box[2]) - float(box[0])
            fh = float(box[3]) - float(box[1])
            scale = min(w / fw, h / fh)
            sw, sh = fw * scale, fh * scale
            x_off = x + (w - sw) / 2
            y_off = y_bot + (h - sh) / 2
            fig_form = pikepdf.Page(fig_page).as_form_xobject()
            fig_form_foreign = pdf.copy_foreign(fig_form)
        resources = target_page.get("/Resources", pikepdf.Dictionary())
        xobjects = resources.get("/XObject", pikepdf.Dictionary())
        counter += 1
        xobj_name = pikepdf.Name(f"/VecFig{counter}")
        xobjects[xobj_name] = fig_form_foreign
        resources[pikepdf.Name("/XObject")] = xobjects
        target_page[pikepdf.Name("/Resources")] = resources
        stamp = (
            f"q {scale:.6f} 0 0 {scale:.6f} {x_off:.2f} {y_off:.2f} cm "
            f"{xobj_name} Do Q"
        )
        new_stream = pdf.make_stream(stamp.encode())
        existing = target_page.get("/Contents")
        if existing is None:
            streams = [new_stream]
        elif isinstance(existing, pikepdf.Array):
            streams = list(existing) + [new_stream]
        else:
            streams = [existing, new_stream]
        target_page[pikepdf.Name("/Contents")] = pikepdf.Array(streams)
    pdf.save()
    pdf.close()


def add_figure_page(c, title, panels, caption=None):
    """Add a page with title, one or more panel images, and optional caption."""
    c.showPage()
    y = PAGE_H - MARGIN

    # Title
    c.setFont("Helvetica-Bold", TITLE_SIZE)
    c.drawString(MARGIN, y, title)
    y -= 20

    if isinstance(panels, list):
        # Multiple panels side by side
        n = len(panels)
        gap = 8
        panel_w = (USABLE_W - gap * (n - 1)) / n
        max_h = PAGE_H - 2 * MARGIN - 40
        max_panel_h = 0
        for i, p in enumerate(panels):
            px = MARGIN + i * (panel_w + gap)
            h = draw_image(c, p, px, y, panel_w, max_h)
            max_panel_h = max(max_panel_h, h)
        y -= max_panel_h + 10
    else:
        # Single panel, full width
        max_h = PAGE_H - 2 * MARGIN - 50
        h = draw_image(c, panels, MARGIN, y, USABLE_W, max_h)
        y -= h + 10

    if caption:
        _draw_caption(c, caption, MARGIN, y, USABLE_W, y_bot=MARGIN)


def add_table_page(c, title, csv_path, caption=None, col_widths=None):
    """Add a page with a table rendered from a CSV file."""
    c.showPage()
    y = PAGE_H - MARGIN

    c.setFont("Helvetica-Bold", TITLE_SIZE)
    c.drawString(MARGIN, y, title)
    y -= 25

    if not csv_path.exists():
        c.setFont("Helvetica-Oblique", 10)
        c.setFillColorRGB(0.6, 0.6, 0.6)
        c.drawString(MARGIN, y - 20, f"[Missing: {csv_path.name}]")
        c.setFillColorRGB(0, 0, 0)
        return

    with open(csv_path) as f:
        rows = list(csv.reader(f))
    if not rows:
        return

    # Auto-compute column widths if not provided
    if col_widths is None:
        n_cols = len(rows[0])
        first_col_w = USABLE_W * 0.40
        remaining = (USABLE_W - first_col_w) / (n_cols - 1)
        col_widths = [first_col_w] + [remaining] * (n_cols - 1)

    table = Table(rows, colWidths=col_widths, repeatRows=1)
    style = TableStyle([
        # Header row
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E5E7EB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        # Body
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),  # first col left
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),  # other cols center
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        # Grid
        ('GRID', (0, 0), (-1, -1), 0.25, colors.HexColor('#D1D5DB')),
        ('LINEBELOW', (0, 0), (-1, 0), 1, colors.black),
        # Alternating row colors
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F9FAFB')]),
        # Bold totals row if present
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold' if 'Total' in rows[-1][0] else 'Helvetica'),
        ('LINEABOVE', (0, -1), (-1, -1), 0.5, colors.black),
        # Padding
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ])
    table.setStyle(style)

    # Compute table height and position
    w, h = table.wrapOn(c, USABLE_W, PAGE_H - 2 * MARGIN - 60)
    table.drawOn(c, MARGIN, y - h)
    y -= h + 15

    if caption:
        _draw_caption(c, caption, MARGIN, y, USABLE_W, y_bot=MARGIN)


def add_stacked_page(c, title, panels_top, panels_bottom, caption=None):
    """Two rows of panels on one page (top/bottom stacked, real heights, tight)."""
    c.showPage()
    y = PAGE_H - MARGIN

    c.setFont("Helvetica-Bold", TITLE_SIZE)
    c.drawString(MARGIN, y, title)
    y -= 20

    half_h = (PAGE_H - 2 * MARGIN - 50) / 2

    # Top row (track actual drawn height)
    if isinstance(panels_top, list):
        n = len(panels_top)
        gap = 8
        pw = (USABLE_W - gap * (n - 1)) / n
        top_h = 0
        for i, p in enumerate(panels_top):
            h = draw_image(c, p, MARGIN + i * (pw + gap), y, pw, half_h)
            top_h = max(top_h, h)
    else:
        top_h = draw_image(c, panels_top, MARGIN, y, USABLE_W, half_h)

    y -= top_h + 8

    # Bottom row
    if isinstance(panels_bottom, list):
        n = len(panels_bottom)
        gap = 8
        pw = (USABLE_W - gap * (n - 1)) / n
        for i, p in enumerate(panels_bottom):
            draw_image(c, p, MARGIN + i * (pw + gap), y, pw, half_h)
    else:
        draw_image(c, panels_bottom, MARGIN, y, USABLE_W, half_h)

    if caption:
        _draw_caption(c, caption, MARGIN, MARGIN + 50, USABLE_W, y_bot=MARGIN)


def main():
    P = PAPER_DIR  # shorthand

    out_path = P / "VoiceFM_figures_v3.pdf"
    c = canvas.Canvas(str(out_path), pagesize=letter)

    # ── Cover page ────────────────────────────────────────────────────
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 2 * inch,
                        "VoiceFM — Paper Figures")
    c.setFont("Helvetica", 12)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 2.5 * inch,
                        "Primary model: VoiceFM-Whisper")
    c.setFont("Helvetica", 10)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 3 * inch,
                        "Cohort: 846 training + 138 validation participants")

    # List tables + figures
    c.setFont("Helvetica", 9)
    figures = [
        "Table 1: Training cohort (n=846)",
        "Table 2: Validation cohort (n=138)",
        "Table 3: Recording battery composition",
        "Figure 1: Architecture + Training Curves",
        "Figure 2: GSD Classification (AUROC bars + per-diagnosis)",
        "Figure 3: Validation Cohort Evaluation",
        "Figure 4: External Transfer + Few-shot",
        "Figure 5: Recording Attribution",
        "Figure 6: PD Detection (NeuroVoz + mPower)",
        "Figure S1: Full Model Comparison (HeAR + others)",
        "Figure S2: Acoustic Grounding (Interpretability)",
        "Figure S3: Embedding Structure",
    ]
    y = PAGE_H - 4.0 * inch
    for fig in figures:
        c.drawString(1.5 * inch, y, fig)
        y -= 16
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(1.5 * inch, y - 10,
                 "All figures use VoiceFM-Whisper as primary model.")

    # ── Table 1: Training cohort ──────────────────────────────────────
    add_table_page(c,
        "Table 1. Training cohort characteristics by disease category",
        P / "table1a_cohort_train_v3.csv",
        "Table 1. Demographic and clinical characteristics of the training cohort (N = 846), "
        "stratified by disease category and used to fit the contrastive VoiceFM model. Continuous "
        "variables (age, PHQ-9, GAD-7, VHI-10) reported as mean ± standard deviation; categorical "
        "variables as N (%). Disease categories are not mutually exclusive at the participant level. "
        "Disease assignment uses clinically-curated gold-standard diagnosis (GSD) criteria."
    )

    # ── Table 2: Validation cohort ────────────────────────────────────
    add_table_page(c,
        "Table 2. Validation cohort characteristics",
        P / "table1b_cohort_test_v3.csv",
        "Table 2. Demographic and clinical characteristics of the validation cohort (N = 138), "
        "enrolled into B2AI Voice after the training cohort was finalized and evaluated only at "
        "inference time (see Figure 3). Same labeling criteria and reporting conventions as Table 1."
    )

    # ── Table 3: Recording battery ────────────────────────────────────
    add_table_page(c,
        "Table 3. Recording battery composition",
        P / "table2_recordings_v3.csv",
        "Table 3. Composition of the recording battery (N = 40,056 recordings; 984 participants) "
        "grouped into 13 task categories. Recordings: number of recordings per category. "
        "Participants: number of participants contributing at least one recording in that category. "
        "Types: number of distinct recording prompts in the category. Mean Duration: average length "
        "per recording in seconds. Total Duration: cumulative time in hours.",
        col_widths=[
            USABLE_W * 0.28,  # Task Category
            USABLE_W * 0.12,  # Recordings
            USABLE_W * 0.14,  # Participants
            USABLE_W * 0.10,  # Types
            USABLE_W * 0.18,  # Mean Duration (s)
            USABLE_W * 0.18,  # Total Duration (h)
        ]
    )

    # ── Figure 1: Architecture + Training ─────────────────────────────
    add_stacked_page(c,
        "Figure 1. VoiceFM architecture and training",
        P / "fig1a_architecture.png",
        P / "fig1bc_training.png",
        "Figure 1. a, VoiceFM dual-encoder architecture: a fine-tuned Whisper large-v2 audio encoder "
        "(layers 28-31 unfrozen) produces 256-dimensional audio embeddings aligned with clinical embeddings "
        "from a tabular transformer over 44 clinical features via symmetric InfoNCE loss, with auxiliary "
        "disease-category BCE and age regression MSE losses. "
        "b, Training loss (train and validation) across epochs for VoiceFM-Whisper (representative seed). "
        "c, Retrieval performance (Recall@5, audio-to-clinical) across epochs. "
        "Trained on 846 participants; early stopping with patience 25."
    )

    # ── Figure 2: GSD Classification ──────────────────────────────────
    add_stacked_page(c,
        "Figure 2. GSD classification performance",
        P / "fig2a_results.png",
        P / "fig2b_diagnoses.png",
        "Figure 2. a, Mean AUROC across five gold-standard diagnosis (GSD) categories for four models — "
        "VoiceFM-Whisper, VoiceFM-HuBERT, Frozen Whisper, Frozen HuBERT. Left group shows the average "
        "across categories; right groups show per-category performance. Error bars: SD across 5 seeds (42–46). "
        "Significance markers: Welch's t-test on the seed-level AUROC distributions. "
        "b, Per-diagnosis AUROC for 9 individual GSD conditions (N ≥ 5 per condition)."
    )

    # ── Figure 3: Prospective evaluation (NEW) ────────────────────────
    add_figure_page(c,
        "Figure 3. Prospective evaluation on held-out cohort",
        P / "fig3_prospective.png",
        "Figure 3. Out-of-sample evaluation on the 138 participants held out from training, enrolled "
        "into B2AI Voice after the training cohort was finalized. "
        "a, Disease-category AUROCs for VoiceFM-Whisper (blue) vs VoiceFM-HuBERT (gray); "
        "5-seed mean ± SD (seeds 42–46). The held-out cohort is fixed across seeds; reported variance "
        "reflects probe sensitivity to the training-cohort split. "
        "b, Per-diagnosis AUROCs for 9 individual GSD conditions, sorted by VoiceFM-Whisper performance. "
        "Cell shading: green ≥ 0.85, yellow 0.70–0.85, red < 0.70."
    )

    # ── Figure 4: External Transfer + Few-shot ────────────────────────
    add_stacked_page(c,
        "Figure 4. External dataset transfer and few-shot learning",
        P / "fig3_transfer.png",
        P / "fig3b_fewshot.png",
        "Figure 4. a, Transfer AUROC on three external voice datasets not seen during training: "
        "Coswara (COVID-19 detection, n=2,098), Saarbrücken Voice Database (SVD; voice pathology, n=2,041), "
        "and MDVR-KCL (Parkinson's disease, n=73). VoiceFM-Whisper (blue) vs Frozen Whisper (gray). "
        "5-fold cross-validation with logistic regression probes on frozen embeddings. Error bars: SD across 5 seeds. "
        "b, Few-shot learning curves (k=1 to 20 labeled examples per class, 100 random trials per k). "
        "Shaded bands show standard deviation of the AUROC distribution: for VoiceFM-Whisper, pooled "
        "across (5 seeds × 100 trials) using the law of total variance (within-trial + between-seed); "
        "for Frozen Whisper (deterministic encoder), across the 100 trials. Both bands therefore "
        "measure sensitivity to support-set sampling and are directly comparable."
    )

    # ── Figure 5: Recording Attribution ───────────────────────────────
    add_figure_page(c,
        "Figure 5. Recording task attribution",
        P / "fig4_attribution.png",
        "Figure 5. a, Per-recording-type AUROC heatmap for the top recording types across the five "
        "disease categories (Control vs Disease, Voice, Neurological, Mood, Respiratory). "
        "Blank cells indicate recording type × disease combinations where AUROC could not be reliably "
        "estimated due to insufficient test participants. b, Greedy forward selection on recording types: "
        "starting from no recordings, iteratively add the task type that most increases mean AUROC across "
        "the five categories. Solid line: 5-seed mean; shaded band: ± SD. The all-types baseline is shown "
        "as a red dashed line. Selection is performed on the same test set used for evaluation, so "
        "absolute peak AUROC may be optimistically biased; relative ranking of task types is the "
        "primary finding."
    )

    # ── Figure 6: PD Detection (composite) ─────────────────────────────
    add_figure_page(c,
        "Figure 6. Application to Parkinson's Disease",
        P / "fig5_pd_combined.png",
        "Figure 6. Application to Parkinson's disease. "
        "Row 1 (panels a–c) shows zero-shot, cross-lingual evaluation on NeuroVoz "
        "(107 participants, Spanish speech; 5-seed mean ± SD throughout). "
        "a, ROC curves for VoiceFM-Whisper (frozen) by task category. The pooled all-tasks ROC is "
        "essentially the speech ROC, consistent with a speech-dominated articulatory signal. "
        "b, Incremental R²: cumulative in-sample variance of VoiceFM P(PD) explained by eGeMAPSv02 "
        "feature groups, in canonical order (F0/jitter/shimmer/HNR → loudness → formants → spectral → MFCC/voicing). "
        "c, Cohen's d effect sizes (PD vs HC) for top 12 eGeMAPSv02 features. Bars are color-coded by "
        "feature class — articulatory (red), spectral (orange), voicing (purple), loudness (blue), "
        "phonatory (gray) — with the legend at the lower right of the panel. "
        "Row 2 (panels d–f) shows fine-tuned evaluation on the mPower sustained-vowel cohort "
        "(VoiceFM-Whisper fine-tune; 585 test participants, 101 PD / 484 control). "
        "d, Test-set ROC (sustained vs countdown vs combined). "
        "e, P(PD) trajectories over 5 months (50 PD vs 50 control participants; thin lines: individual "
        "trajectories, thick lines: monthly group means). "
        "f, Incremental R² on mPower sustained vowels. "
        "g, Cohen's d for top 12 mPower eGeMAPSv02 features; formant frequencies and bandwidths "
        "dominate. Features marked † are sex-confounded and not significant within sex; arrows ↑/↓ on "
        "labels indicate the direction of the PD−HC difference."
    )

    # ── Supplementary Figures ─────────────────────────────────────────
    add_figure_page(c,
        "Figure S1. Full model comparison and demographic baseline",
        P / "figS1_models.png",
        "Figure S1. GSD category AUROC for six audio-embedding models — VoiceFM-Whisper, VoiceFM-HuBERT, "
        "VoiceFM-HeAR, Frozen Whisper, Frozen HuBERT, Frozen HeAR — and a demographics-only baseline "
        "(black bars: logistic regression on age + sex alone, no audio). VoiceFM-HeAR uses a frozen "
        "Google HeAR encoder with only the projection layer trainable. The demographic baseline "
        "is shown to rule out the possibility that AUROC is driven by demographic confounding rather "
        "than acoustic disease information. Error bars: standard deviation across 5 seeds (42–46)."
    )

    add_figure_page(c,
        "Figure S2. Acoustic grounding (interpretability)",
        P / "figS2_interpretability.png",
        "Figure S2. a, Ridge regression R² (5-fold CV, scaler fit inside fold) for 14 classical acoustic "
        "features decoded from VoiceFM-Whisper (blue) vs Frozen Whisper (gray) embeddings. "
        "b, Spearman rank correlation between each acoustic feature and the top PCA components of "
        "VoiceFM-Whisper embeddings. PC1 separates healthy-voice markers (CPPS, HNR) from "
        "pathological-voice markers (jitter, shimmer)."
    )

    add_figure_page(c,
        "Figure S3. Embedding structure",
        P / "figS3_embedding_structure.png",
        "Figure S3. a, Nearest-neighbor retrieval (k=5, cosine distance) on the 846 training participants: "
        "fraction of NNs sharing the query participant's GSD category, for VoiceFM-Whisper vs Frozen Whisper. "
        "The chance baseline (random pair sharing a category) is ≈0.24. Welch's t-test on the per-participant "
        "match rates is significant (p = 3.0×10⁻⁵), and the effect is robust across k = 1, 3, 10 "
        "(all p < 10⁻⁴). Error bars: 95% CI of the mean. "
        "b, Within-participant consistency for VoiceFM-Whisper: mean cosine similarity between embeddings "
        "of different recording types from the same participant (intra-person) vs different participants "
        "(inter-person). Bars show per-participant means; error bars show ±1 SD across the 845 training "
        "participants who have ≥2 recordings."
    )

    # ── Save ──────────────────────────────────────────────────────────
    c.save()
    n_vector = len(PDF_OVERLAYS)
    if n_vector:
        stamp_vector_overlays(out_path)
    print(f"Saved: {out_path}")
    print(f"Pages: cover + 3 tables + 6 main figures + 3 supplementary figures")
    print(f"Vector overlays applied: {n_vector} panels")


if __name__ == "__main__":
    main()
