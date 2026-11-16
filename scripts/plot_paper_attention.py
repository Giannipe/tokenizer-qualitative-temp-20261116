#!/usr/bin/env python
"""Compact paper-style DINO/AFA comparisons and fixed-query temporal panels."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image

from plot_tokenizer_qualitative import COLORS, probabilities


INK, MUTED = "#17212e", "#667085"
TASKS = {1: "Cube placement", 2: "Plush placement", 3: "Tool-assisted pushing"}
FRONT_SLOTS = [2, 5, 6, 4]
WRIST_SLOTS = [6, 9, 13, 4]


def upsample(x, size=256):
    return np.asarray(Image.fromarray(np.asarray(x, dtype=np.float32)).resize((size, size), Image.Resampling.BILINEAR))


def panel(ax, rgb, distribution=None, title=None):
    ax.imshow(np.clip(rgb, 0, 1), interpolation="nearest")
    if distribution is not None:
        p = np.asarray(distribution, dtype=float)
        p = p / max(p.sum(), 1e-12)
        lift = np.clip(np.log2(np.maximum(p*p.size, 1)), 0, 3)
        strength = upsample(lift) / 3
        rgba = plt.get_cmap("magma")(strength)
        rgba[..., 3] = .65*strength
        ax.imshow(rgba, interpolation="nearest")
    ax.set(xlim=(-.5, 255.5), ylim=(255.5, -.5))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=7, color=INK, pad=4)


def row_label(ax, label, color=INK):
    color = np.asarray(matplotlib.colors.to_rgb(color))
    if color @ [.2126, .7152, .0722] > .6:
        color = color * .65
    ax.text(-.06, .5, label, transform=ax.transAxes, ha="right", va="center", fontsize=8, color=color)


def colorbar(fig, rect=(.35, .05, .4, .012)):
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 3), cmap="magma"),
        cax=fig.add_axes(rect), orientation="horizontal", ticks=[0, 1, 2, 3])
    cb.ax.set_xticklabels(["≤1×", "2×", "4×", "≥8×"])
    cb.ax.tick_params(labelsize=7, length=2, pad=2)
    cb.set_label("Spatial attention relative to uniform · shared scale", fontsize=7, labelpad=3)
    cb.outline.set_linewidth(.5)


class Study:
    def __init__(self, root, baseline, output, dpi):
        self.root, self.output, self.dpi = root, output, dpi
        self.manifest = json.loads((root/"manifest.json").read_text())
        self.records = self.manifest["records"]
        self.by_id = {r["id"]: r for r in self.records}
        self.arrays = {r["id"]: dict(np.load(root/"arrays"/f"{r['id']}.npz", allow_pickle=False)) for r in self.records}
        self.baseline = None if baseline is None else {
            r["id"]: dict(np.load(baseline/"arrays"/f"{r['id']}.npz", allow_pickle=False)) for r in self.records}
        self.figures = []

    def comparison_records(self, second=False):
        result = []
        for task in (1, 2, 3):
            candidates = [r for r in self.records if r["task_id"] == task and r["selection_origin"] == "relative_time"]
            episode = sorted({r["episode"] for r in candidates})[int(second)]
            result += [min([r for r in candidates if r["episode"] == episode], key=lambda r: abs(r["fraction"]-fraction))
                       for fraction in (.3, .7)]
        return result

    def save(self, fig, name, caption, records, slots, family):
        base = self.output/family/name
        base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(base.with_suffix(".png"), dpi=self.dpi, bbox_inches="tight", facecolor="white")
        fig.savefig(base.with_suffix(".pdf"), dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        self.figures.append(dict(name=name, family=family, caption=caption,
            records=[r["id"] for r in records], slots=slots,
            png=str(base.with_suffix(".png").relative_to(self.output)), pdf=str(base.with_suffix(".pdf").relative_to(self.output))))
        print(f"Saved {family}/{name}", flush=True)


def comparison(study, view, second, name, family):
    records = study.comparison_records(second)
    slots = FRONT_SLOTS if view == 0 else WRIST_SLOTS
    fig, axes = plt.subplots(6, 6, figsize=(10.5, 10.8))
    fig.subplots_adjust(left=.13, right=.99, top=.88, bottom=.10, wspace=.035, hspace=.06)
    camera = "Front" if view == 0 else "Wrist"
    fig.text(.13, .975, f"Dense backbone attention and learned AFA aggregation · {camera.lower()}", fontsize=12, color=INK, va="top")
    fig.text(.13, .945, "Same checkpoint, observation and instruction · verified demonstrations", fontsize=8, color=MUTED)
    for task, col in zip((1, 2, 3), (0, 2, 4)):
        x = (axes[0, col].get_position().x0 + axes[0, col+1].get_position().x1)/2
        fig.text(x, .912, TASKS[task], fontsize=9, color=INK, ha="center")
    for col, r in enumerate(records):
        d = study.arrays[r["id"]]
        rgb = d[f"rgb_{view}"]
        panel(axes[0, col], rgb, title=f"ep {r['episode']} · {r['frame']/r['fps']:.2f} s")
        panel(axes[1, col], rgb, d[f"dino_cls_{view}"])
        axes[1, col].text(.97, .04, f"patch mass {d[f'dino_cls_patch_mass_{view}'].mean():.0%}",
            transform=axes[1, col].transAxes, ha="right", color="white", fontsize=5.5,
            bbox=dict(fc=INK, ec="none", alpha=.8, pad=1.5))
        for row, slot in enumerate(slots, 2):
            panel(axes[row, col], rgb, d[f"attention_{view}"][slot])
    row_label(axes[0, 0], "RGB")
    row_label(axes[1, 0], "DINO\nCLS → patch")
    for row, slot in enumerate(slots, 2):
        row_label(axes[row, 0], f"AFA\nS{slot:02d}", COLORS[slot])
    colorbar(fig)
    caption = (f"{camera} camera: last-layer DINO CLS-to-patch attention and four AFA spatial pooling maps from the same "
        "fine-tuned early-BERT checkpoint. Each column uses the same input in every row. The two observations per task "
        "are at 30% and 70% of an episode; they are not phase annotations. CLS attention is averaged across heads, "
        "CLS/register keys are excluded, and the remaining patch weights are normalized spatially. Every map uses the "
        "same attention-relative-to-uniform scale, with values at/below uniform transparent and values above 8× saturated. "
        "The fixed slot IDs illustrate different aggregation patterns; they do not denote object categories. This is a "
        "within-checkpoint representation diagnostic, not a comparison of policy success or a causal saliency analysis.")
    study.save(fig, name, caption, records, slots, family)


def temporal(study, sequence, view, slots, name, family):
    seq = study.manifest["sequences"][sequence]
    records = [study.by_id[ident] for ident in seq["record_ids"]]
    fig, axes = plt.subplots(len(slots)+1, len(records), figsize=(7.2, 8.3), squeeze=False)
    fig.subplots_adjust(left=.15, right=.99, top=.88, bottom=.12, wspace=.035, hspace=.08)
    camera = "Front" if view == 0 else "Wrist"
    fig.text(.15, .975, f"Learned spatial pooling through time · {TASKS[records[0]['task_id']]}", fontsize=10.5, color=INK, va="top")
    fig.text(.15, .94, f"{camera} camera · same four queries across time · verified demonstration ep {records[0]['episode']}", fontsize=7.5, color=MUTED)
    for col, r in enumerate(records):
        d = study.arrays[r["id"]]
        rgb = d[f"rgb_{view}"]
        panel(axes[0, col], rgb, title=f"{seq['stages'][col]}\n{r['frame']/r['fps']:.2f} s")
        for row, slot in enumerate(slots, 1):
            panel(axes[row, col], rgb, d[f"attention_{view}"][slot])
    row_label(axes[0, 0], "RGB")
    for row, slot in enumerate(slots, 1):
        row_label(axes[row, 0], f"S{slot:02d}", COLORS[slot])
    colorbar(fig)
    caption = (f"Temporal visualization of four fixed {camera.lower()} AFA queries, before the AFA transformer. "
        "The top row shows RGB; following rows show per-query spatial pooling distributions. Frames were selected "
        "by inspecting RGB interactions, independently of DINO CLS attention. Stage labels are descriptive readings "
        "of the images, not ground-truth phase annotations. The same slot IDs and attention scale are retained at every "
        "time step, including diffuse or redundant behavior. Stable query identity does not establish object tracking. "
        "The demonstration is a recorded training example, not a measured closed-loop success of the policy.")
    study.save(fig, name, caption, records, slots, family)


def routing_rgb(study):
    records = []
    for task in (1, 2, 3):
        records.append(next(r for r in study.records if r["task_id"] == task and r["selection_origin"] == "relative_time"
            and abs(r["fraction"]-.5)<1e-6 and r["episode"] == {1:0, 2:50, 3:100}[task]))
    slots = FRONT_SLOTS
    fig, axes = plt.subplots(3, 6, figsize=(10.5, 6))
    fig.subplots_adjust(left=.13, right=.99, top=.85, bottom=.11, wspace=.035, hspace=.10)
    fig.text(.13, .975, "RGB responsibility views of learned slots", fontsize=12, color=INK, va="top")
    fig.text(.13, .93, "Competitive routing applied to RGB · illustrative masks, without a reconstruction decoder", fontsize=8, color=MUTED)
    for row, r in enumerate(records):
        d = study.arrays[r["id"]]
        rgb = d["rgb_0"]
        routing = d["routing_0"].reshape(16, 16, 16)
        routing = routing / routing.sum(0, keepdims=True)
        labels = routing.argmax(0)
        colors = COLORS[labels]
        colors = np.asarray(Image.fromarray((colors*255).astype(np.uint8)).resize((256,256), Image.Resampling.NEAREST))/255
        panel(axes[row, 0], rgb, title="RGB" if row == 0 else None)
        panel(axes[row, 1], .45*rgb+.55*colors, title="Winning query" if row == 0 else None)
        for col, slot in enumerate(slots, 2):
            weight = upsample(routing[slot])[..., None]
            panel(axes[row, col], rgb*weight+(1-weight), title=f"S{slot:02d}" if row == 0 else None)
        row_label(axes[row, 0], TASKS[r["task_id"]].replace(" ", "\n", 1))
    caption = ("Slot-Attention-inspired layout using actual competitive-routing responsibilities. The combined color "
        "panel shows the winning query per patch; each individual panel shows R(k|patch) × RGB + (1-R(k|patch)) × white. "
        "There is no independent mask contrast normalization. These are routing-weighted input views, not decoded "
        "reconstructions, ground-truth object masks, or demonstrated instance segmentations. Native routing resolution "
        "is 16×16; interpolated masks and fixed query colors are used for display.")
    study.save(fig, "A05_routing_weighted_rgb", caption, records, slots, "appendix")


def policy_comparison(study, backbone=False):
    if study.baseline is None:
        return
    records = study.comparison_records(False)
    fig, axes = plt.subplots(3, 6, figsize=(10.5, 5.8))
    fig.subplots_adjust(left=.15, right=.99, top=.84, bottom=.15, wspace=.035, hspace=.08)
    title = "Backbone attention in two fine-tuned policies" if backbone else "Two fine-tuned policies on identical recorded inputs"
    subtitle = "Front camera · same last-layer CLS-to-patch diagnostic in both encoders" if backbone else "Front camera · final ACT cross-attention · conditional spatial distributions"
    fig.text(.15, .975, title, fontsize=12, color=INK, va="top")
    fig.text(.15, .93, subtitle, fontsize=8, color=MUTED)
    for col, r in enumerate(records):
        d, b = study.arrays[r["id"]], study.baseline[r["id"]]
        np.testing.assert_allclose(d["rgb_0"], b["rgb_0"], atol=1e-7)
        panel(axes[0, col], d["rgb_0"], title=f"{TASKS[r['task_id']]}\n{r['frame']/r['fps']:.2f} s")
        key = "dino_cls_0" if backbone else "act_map_0"
        panel(axes[1, col], b["rgb_0"], b[key])
        panel(axes[2, col], d["rgb_0"], d[key])
        for row, values in ((1, b), (2, d)):
            mass = float(values['dino_cls_patch_mass_0'].mean()) if backbone else float(values['act_view_mass_0'])
            axes[row,col].text(.97,.04,f"{'patch' if backbone else 'view'} mass {mass:.0%}",
                transform=axes[row,col].transAxes,ha="right",color="white",fontsize=5.5,
                bbox=dict(fc=INK,ec="none",alpha=.8,pad=1.5))
    for ax, label in zip(axes[:, 0], ["RGB", "TurboVLA\nVLI", "TurboVLA-AFA\nearly-BERT"]):
        row_label(ax, label)
    colorbar(fig, (.35,.06,.4,.015))
    if backbone:
        caption = ("The same last-layer DINO CLS-to-patch diagnostic, averaged over 12 heads and normalized over "
            "patch positions after excluding the CLS/register keys, in the separately fine-tuned TurboVLA and "
            "TurboVLA-AFA checkpoints. Patch mass before spatial renormalization is shown separately. Identical "
            "RGB, instructions and states are replayed in both policies, with the same display scale. These are "
            "backbone self-attention distributions, not action-to-image attention or a causal tokenizer ablation. "
            "The encoders were trained as parts of different architectures (VLI versus early-BERT/AFA).")
        study.save(fig, "A07_backbone_attention_comparison", caption, records, [], "appendix")
        return
    caption = ("Same observations, instructions and robot states replayed through separately fine-tuned TurboVLA "
        "and TurboVLA-AFA. Final-layer ACT cross-attention is averaged over eight heads and 12 action queries. "
        "TurboVLA weights are displayed on their dense visual token positions after VLI; AFA weights are projected "
        "through the tokenizer's spatial maps, preceding the AFA transformer. Both are normalized conditionally on "
        "the displayed view; the absolute view mass is shown separately. The architectures also differ in VLI versus "
        "early-BERT interaction and potentially fine-tuned backbone weights. This is a full-policy diagnostic comparison, "
        "not an isolated tokenizer ablation or a causal pixel-attribution comparison.")
    study.save(fig, "A06_policy_attention_comparison", caption, records, [], "appendix")


def write_docs(study):
    study.output.mkdir(parents=True, exist_ok=True)
    (study.output/"figure_manifest.json").write_text(json.dumps(dict(figures=study.figures,
        normalized_scale="log2 attention relative to uniform; clip [0,3]; alpha=0.65*value/3",
        front_slots=FRONT_SLOTS,wrist_slots=WRIST_SLOTS,dpi=study.dpi,
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    lines = ["# Paper-style qualitative figures", "", "Two main-text candidates; supporting comparisons in the appendix.", ""]
    for family in ("main","appendix"):
        lines += [f"## {family.title()}", ""]
        for f in study.figures:
            if f["family"] != family:
                continue
            lines += [f"### {f['name']}", "", f"[PDF]({f['pdf']}) · [PNG]({f['png']})", "", f["caption"], "",
                      f"![{f['name']}]({f['png']})", ""]
    (study.output/"GALLERY.md").write_text("\n".join(lines))
    captions = ["# Figure captions", ""]
    for f in study.figures:
        captions += [f"## {f['name']}", "", f["caption"], ""]
    (study.output/"CAPTIONS.md").write_text("\n".join(captions))
    cards = "".join(f'<article><h2>{f["name"]}</h2><a href="{f["pdf"]}">PDF</a><img src="{f["png"]}" loading="lazy"><p>{f["caption"]}</p></article>' for f in study.figures)
    (study.output/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>Paper-style AFA figures</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;color:#17212e}img{width:100%}article{margin:50px 0}p{line-height:1.6;color:#667085}</style>'
        '<h1>Paper-style AFA diagnostics</h1>'+cards)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("afa", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"pdf.fonttype":42,"ps.fonttype":42})
    study = Study(args.afa,args.baseline,args.output,args.dpi)
    comparison(study,0,False,"M01_dino_to_afa","main")
    temporal(study,"tool_demo",1,WRIST_SLOTS,"M02_token_attention_through_time","main")
    comparison(study,1,False,"A01_dino_to_afa_wrist","appendix")
    comparison(study,0,True,"A02_dino_to_afa_second_episodes","appendix")
    temporal(study,"cube_demo",0,FRONT_SLOTS,"A03_cube_temporal_attention","appendix")
    temporal(study,"plush_demo",0,FRONT_SLOTS,"A04_plush_temporal_attention","appendix")
    routing_rgb(study)
    policy_comparison(study)
    policy_comparison(study, backbone=True)
    write_docs(study)


if __name__ == "__main__":
    main()
