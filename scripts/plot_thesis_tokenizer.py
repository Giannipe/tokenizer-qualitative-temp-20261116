#!/usr/bin/env python
"""Curated, publication-size AFA figures from an explicit RGB frame selection.

All spatial maps use one fixed attention/uniform scale. Slot IDs never change
within a sequence. Query-to-word panels show actual probabilities, including
special tokens, rather than text labels inferred from image regions.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_rgb
import numpy as np
from PIL import Image

from plot_tokenizer_qualitative import COLORS, probabilities


INK = "#17212e"
MUTED = "#667085"
GREEN = "#257561"
RED = "#a43f4a"
VIEWS = ("Front", "Wrist")


def upsample(x, size=256):
    return np.asarray(Image.fromarray(x.astype(np.float32)).resize(
        (size, size), Image.Resampling.BILINEAR))


def stats(data, view, slot):
    p = probabilities(data[f"attention_{view}"])[slot].reshape(16, 16)
    entropy = float(-(p * np.log(np.maximum(p, 1e-12))).sum() / np.log(p.size))
    padding = float(p[:2].sum() + p[14:].sum())
    return p, entropy, padding


def panel(ax, data, view=0, slot=None, title=None, entropy=False):
    ax.imshow(data[f"rgb_{view}"], interpolation="nearest")
    if slot is not None:
        p, h, _ = stats(data, view, slot)
        lift = np.clip(np.log2(np.maximum(p * p.size, 1)), 0, 3)
        strength = upsample(lift) / 3
        overlay = plt.get_cmap("magma")(strength)
        overlay[..., 3] = .62 * strength
        ax.imshow(overlay, interpolation="nearest")
        xy = (data[f"keypoints_{view}"][slot] * .9375 + 1) * 128
        ax.scatter(*xy, s=10, c=[COLORS[slot]], edgecolors="white", linewidths=.55, zorder=4)
        if entropy:
            ax.text(.96, .04, f"H={h:.2f}", transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=6.5, color="white", bbox=dict(fc=INK, ec="none", alpha=.85, pad=1.8))
    ax.set(xlim=(-.5, 255.5), ylim=(255.5, -.5))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=8, pad=4, color=INK)


def row_label(ax, text, color=INK):
    rgb = np.asarray(to_rgb(color))
    if np.dot(rgb, [.2126, .7152, .0722]) > .55:
        color = rgb * .65
    ax.text(-.06, .5, text, transform=ax.transAxes, ha="right", va="center",
            fontsize=8, color=color, linespacing=1.5)


def colorbar(fig, rect=(.35, .045, .40, .012), label=True):
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 3), cmap="magma"),
                      cax=fig.add_axes(rect), orientation="horizontal", ticks=[0, 1, 2, 3])
    cb.ax.set_xticklabels(["1×", "2×", "4×", "≥8×"])
    cb.ax.tick_params(labelsize=7, length=2, pad=2)
    cb.outline.set_linewidth(.5)
    if label:
        cb.set_label("Spatial attention relative to uniform", fontsize=7, labelpad=3)


def heading(fig, title, subtitle):
    fig.text(.15, .975, title, ha="left", va="top", fontsize=11, color=INK, fontweight="medium")
    fig.text(.15, .94, subtitle, ha="left", va="top", fontsize=7.5, color=MUTED)


class Study:
    def __init__(self, root, output, dpi):
        self.root, self.output, self.dpi = root, output, dpi
        self.selection = json.loads((root / "selection.json").read_text())
        self.records = {r["id"]: r for r in self.selection["records"]}
        self.arrays = {ident: dict(np.load(root / "analysis/arrays" / f"{ident}.npz", allow_pickle=False))
                       for ident in self.records}
        self.sequences = {s["name"]: s for s in self.selection["sequences"]}
        self.figures = []

    def sequence(self, name):
        return [self.arrays[i] for i in self.sequences[name]["record_ids"]]

    def times(self, name):
        return [self.records[i]["frame"] / self.records[i]["fps"]
                for i in self.sequences[name]["record_ids"]]

    def save(self, fig, family, name, title, caption, sequences, slots, view):
        directory = self.output / family
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name.rstrip(".")
        fig.savefig(path.with_suffix(".png"), dpi=self.dpi, bbox_inches="tight", facecolor="white")
        fig.savefig(path.with_suffix(".pdf"), dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        self.figures.append(dict(id=name.rstrip("."), family=family, title=title, caption=caption,
                                 png=str(path.with_suffix(".png").relative_to(self.output)),
                                 pdf=str(path.with_suffix(".pdf").relative_to(self.output)),
                                 sequences=sequences, slots=slots, view=view))
        print(f"Saved {family}/{name}", flush=True)


def temporal(study, sequence, slots, view, family, name, title, caption):
    data = study.sequence(sequence)
    times = study.times(sequence)
    fig, axes = plt.subplots(4, 4, figsize=(7.2, 7.5))
    fig.subplots_adjust(left=.15, right=.99, top=.89, bottom=.12, wspace=.035, hspace=.07)
    heading(fig, title, f"{VIEWS[view]} camera · fixed query IDs across time · recorded observation replay")
    for col, (d, t) in enumerate(zip(data, times)):
        panel(axes[0, col], d, view, title=f"t = {t:.2f} s")
        for row, slot in enumerate(slots, 1):
            panel(axes[row, col], d, view, slot, entropy=True)
    row_label(axes[0, 0], "RGB")
    for row, slot in enumerate(slots, 1):
        row_label(axes[row, 0], f"S{slot:02d}", COLORS[slot])
    colorbar(fig)
    study.save(fig, family, name, title, caption, [sequence], slots, view)


def target_selection(study):
    fig, axes = plt.subplots(4, 4, figsize=(7.2, 7.8))
    fig.subplots_adjust(left=.17, right=.99, top=.89, bottom=.13, wspace=.035, hspace=.13)
    heading(fig, "Target selection: demonstration and failed rollout", 'Instruction: “Grab the plush and place in the box”')
    for block, (sequence, group, color) in enumerate([
        ("plush_demo", "Verified\ndemonstration", GREEN),
        ("plush_failure", "Failed\npolicy rollout", RED),
    ]):
        row = block * 2
        for col, (d, t) in enumerate(zip(study.sequence(sequence), study.times(sequence))):
            panel(axes[row, col], d, 0, title=f"t = {t:.2f} s")
            panel(axes[row + 1, col], d, 1, 6, entropy=True)
        row_label(axes[row, 0], f"{group}\nFront RGB", color)
        row_label(axes[row + 1, 0], "Wrist\nS06", COLORS[6])
    colorbar(fig)
    caption = ("Recorded target-selection contrast: verified demonstration episode 50 and failed rollout episode 0. "
               "The instruction asks for the plush; the verified demonstration "
               "moves the plush toward the box, whereas the illustrated failed rollout transports the red cube. "
               "Each block shows front RGB above the spatial attention of the same wrist query S06. Frames were "
               "chosen by inspecting visible interactions, at the timestamps shown; they are not matched initial states. "
               "All maps use the same attention/uniform scale. This illustrates that query identity and language "
               "conditioning do not guarantee object-category-consistent spatial pooling. It does not establish "
               "that this query caused the wrong-object action. Demonstrations are recorded training examples, "
               "not closed-loop successes measured in this analysis.")
    study.save(fig, "main", "M01_target_selection.", "Target selection across recorded episodes", caption,
               ["plush_demo", "plush_failure"], [6], "front RGB + wrist attention")


def redundancy(study):
    d = study.sequence("cube_detail")[0]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.3))
    fig.subplots_adjust(left=.06, right=.98, top=.86, bottom=.16, wspace=.06, hspace=.20)
    fig.text(.06, .975, "Focused, diffuse and overlapping slots", fontsize=11,
             color=INK, va="top")
    fig.text(.06, .925, "Cube-placement rollout · episode 0 · front camera · t = 3.60 s", fontsize=7.5, color=MUTED)
    panel(axes[0, 0], d, 0, title="RGB observation")
    panel(axes[0, 1], d, 0, title="All 16 slot means")
    for slot in range(16):
        xy = (d['keypoints_0'][slot] * .9375 + 1) * 128
        axes[0, 1].scatter(*xy, c=[COLORS[slot]], s=10, edgecolors="white", linewidths=.5)
    for ax, slot, label in [(axes[0, 2], 0, "S00"), (axes[1, 0], 1, "S01"),
                             (axes[1, 1], 4, "S04"), (axes[1, 2], 5, "S05")]:
        panel(ax, d, 0, slot, title=label, entropy=True)
    colorbar(fig, rect=(.35, .06, .40, .015))
    study.save(fig, "main", "M03_pooling_diversity.", "Concentrated, diffuse and overlapping pooling",
               "Illustrative spatial pooling patterns in one observation. S00 and S01 attend to similar robot regions, "
               "S04 is diffuse, and S05 is more concentrated. All 16 attention-weighted means are shown for context. "
               "The four slots were selected to expose overlap, diffusion and concentration, rather than as a random "
               "sample or evidence for object-instance segmentation. H is normalized spatial entropy. Full 16-slot "
               "coverage is provided in the appendix; black image padding remains visible and is included in all statistics.",
               ["cube_detail"], [0, 1, 4, 5], 0)


def language_pair(study, sequence, slots, view, indices, name, title):
    data = study.sequence(sequence)
    times = study.times(sequence)
    selected = [data[i] for i in indices]
    words = data[0]["words"].tolist()
    language = data[0][f"language_attention_{view}"].mean(0)[slots]
    fig = plt.figure(figsize=(7.2, 6.4))
    grid = fig.add_gridspec(len(slots), 4, width_ratios=[1, 1, 1, 1.65],
                           left=.085, right=.97, top=.85, bottom=.22, wspace=.07, hspace=.07)
    fig.text(.085, .975, title, fontsize=11, color=INK, va="top")
    task = study.records[study.sequences[sequence]['record_ids'][0]]["task"]
    fig.text(.085, .935, task, fontsize=8, color=MUTED)
    fig.text(.085, .895, f"{VIEWS[view]} spatial attention across time", fontsize=8, color=INK)
    for row, slot in enumerate(slots):
        for col, (d, idx) in enumerate(zip(selected, indices)):
            ax = fig.add_subplot(grid[row, col])
            panel(ax, d, view, slot, title=f"{times[idx]:.2f} s" if row == 0 else None, entropy=True)
            if col == 0:
                row_label(ax, f"S{slot:02d}", COLORS[slot])
    ax = fig.add_subplot(grid[:, 3])
    im = ax.imshow(language, cmap="cividis", vmin=0, vmax=.5, aspect="auto", interpolation="nearest")
    ax.set_title("Query → BERT tokens", fontsize=8, pad=8)
    ax.set_xticks(range(len(words)), words, rotation=65, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(slots)), [f"S{k:02d}" for k in slots], fontsize=7)
    ax.tick_params(axis="both", length=0, pad=3)
    for row in range(len(slots)):
        for col in range(len(words)):
            value = language[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=4.6,
                    color="white" if value < .23 else INK)
    cb = fig.colorbar(im, cax=ax.inset_axes([0, -.31, 1, .025]), orientation="horizontal", ticks=[0, .25, .5])
    cb.ax.tick_params(labelsize=6, length=2)
    cb.set_label("Mean over 12 heads · absolute probability", fontsize=6.5)
    colorbar(fig, rect=(.16, .07, .34, .012))
    caption = (f"Spatial–language correspondence for {VIEWS[view].lower()} queries "
               + ", ".join(f"S{k:02d}" for k in slots)
               + ". Rows align the same fixed query across three selected frames with its early-BERT attention "
               "distribution. All BERT tokens, including [CLS] and [SEP], are retained; values are absolute "
               "probabilities averaged over 12 heads, displayed on a common [0, 0.5] scale and rounded to two decimals. "
               "The spatial maps use a shared log attention/uniform scale. In this checkpoint the early-BERT query "
               "attention is computed before image pooling, so it is invariant across these frames for the same "
               "instruction and camera; it is shown once. A high weight on an object word is not evidence that the "
               "spatial map segments that object. Selection is illustrative, not an alignment-accuracy evaluation.")
    study.save(fig, "appendix", name, title, caption, [sequence], slots, view)


def full_language(study):
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.4))
    fig.subplots_adjust(left=.09, right=.95, top=.90, bottom=.14, wspace=.18, hspace=.68)
    fig.text(.09, .975, "All query-to-word distributions", fontsize=11, color=INK, va="top")
    fig.text(.09, .943, "16 queries · all instruction tokens · head-averaged probabilities", fontsize=7.5, color=MUTED)
    for row, sequence in enumerate(["cube_demo", "plush_demo", "tool_demo"]):
        d = study.sequence(sequence)[0]
        for view in (0, 1):
            ax = axes[row, view]
            im = ax.imshow(d[f"language_attention_{view}"].mean(0), cmap="cividis", vmin=0, vmax=.5,
                           aspect="auto", interpolation="nearest")
            ax.set_title(f"Task {row + 1} · {VIEWS[view]}", loc="left", fontsize=8)
            ax.set_xticks(range(len(d["words"])), d["words"], rotation=55, ha="right", fontsize=6.5)
            ax.set_yticks([0, 5, 10, 15], ["S00", "S05", "S10", "S15"], fontsize=7)
            ax.tick_params(length=0, pad=3)
    cb = fig.colorbar(im, cax=fig.add_axes([.30, .055, .40, .012]), orientation="horizontal", ticks=[0, .25, .5])
    cb.ax.tick_params(labelsize=7)
    cb.set_label("Mean over 12 heads · absolute probability", fontsize=7)
    study.save(fig, "appendix", "A04_all_query_word_maps.", "All query-to-word distributions",
               "Complete early-BERT query-to-word attention for all 16 queries, three tasks and two cameras. "
               "Rows are query IDs and columns are the actual BERT tokens, including special tokens. Every panel "
               "uses the same absolute-probability color range. Poolers and early-BERT modules are separate across "
               "cameras; matching IDs across front and wrist do not imply matched semantic roles.",
               ["cube_demo", "plush_demo", "tool_demo"], list(range(16)), [0, 1])


def atlas(study, sequence, index, view, name):
    d = study.sequence(sequence)[index]
    t = study.times(sequence)[index]
    fig, axes = plt.subplots(4, 4, figsize=(7.2, 7.8))
    fig.subplots_adjust(left=.04, right=.99, top=.88, bottom=.13, wspace=.05, hspace=.19)
    fig.text(.04, .975, "Complete spatial attention atlas", fontsize=11, color=INK, va="top")
    fig.text(.04, .938, f"{sequence.replace('_', ' ')} · {VIEWS[view]} · t = {t:.2f} s · all 16 queries", fontsize=8, color=MUTED)
    for slot, ax in enumerate(axes.flat):
        panel(ax, d, view, slot, title=f"S{slot:02d}", entropy=True)
    colorbar(fig)
    study.save(fig, "appendix", name, "Complete spatial attention atlas",
               "All 16 spatial attention maps for one selected observation, using the same RGB, interpolation "
               "and attention/uniform scale as the main figures. No query is omitted. H denotes normalized spatial "
               "entropy over all 256 patches, including the black padding. Query labels denote learned identities, "
               "not annotated object classes.", [sequence], list(range(16)), view)


def write_artifacts(study):
    rows = []
    for ident, data in study.arrays.items():
        record = study.records[ident]
        for view in (0, 1):
            for slot in range(16):
                _, entropy, padding = stats(data, view, slot)
                words = data["words"]
                lang = data[f"language_attention_{view}"].mean(0)[slot]
                rows.append(dict(observation=ident, sequence=record["selection_sequence"],
                                 camera=VIEWS[view].lower(), slot=slot, normalized_entropy=entropy,
                                 attention_mass_on_padding=padding,
                                 strongest_word=str(words[lang.argmax()]), strongest_word_weight=float(lang.max())))
    with (study.output / "slot_metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    consistency = {}
    for name, seq in study.sequences.items():
        data = study.sequence(name)
        for view in (0, 1):
            reference = data[0][f"language_attention_{view}"]
            consistency[f"{name}_{VIEWS[view].lower()}"] = max(
                float(np.abs(d[f"language_attention_{view}"] - reference).max()) for d in data)
    same_instruction = {}
    for view in (0, 1):
        same_instruction[VIEWS[view].lower()] = float(np.abs(
            study.sequence("plush_demo")[0][f"language_attention_{view}"]
            - study.sequence("plush_failure")[0][f"language_attention_{view}"]).max())
    manifest = dict(figures=study.figures, attention_scale="log2 lift over uniform, clipped to [0,3]",
                    spatial_overlay_opacity="0.62 * displayed_log2_lift / 3", dpi=study.dpi,
                    preserves_full_preprocessed_rgb_including_padding=True,
                    language_scale=[0, .5], language_frame_max_abs_difference=consistency,
                    same_instruction_demo_failure_language_max_abs_difference=same_instruction,
                    renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (study.output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2))
    text = ["# Thesis figure captions", "", "These are purposively selected illustrative cases, not random samples.", ""]
    for family, title in [("main", "Main-text candidates"), ("appendix", "Appendix")]:
        text += [f"## {title}", ""]
        for f in study.figures:
            if f['family'] == family:
                text += [f"### {f['id']} — {f['title']}", "", f"[PDF]({f['pdf']}) · [PNG]({f['png']})", "", f['caption'], ""]
    (study.output / "CAPTIONS.md").write_text("\n".join(text))
    sections = []
    for family in ("main", "appendix"):
        items = "".join(f'<article><h3>{f["id"]} · {f["title"]}</h3><a href="{f["pdf"]}">PDF</a> · '
                        f'<a href="{f["png"]}">PNG</a><a href="{f["png"]}"><img loading="lazy" '
                        f'src="{f["png"]}"></a><p>{f["caption"]}</p></article>'
                        for f in study.figures if f['family'] == family)
        sections.append(f'<h2>{family.title()}</h2>{items}')
    (study.output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Thesis AFA figures</title>'
        '<style>body{font:16px system-ui;max-width:1000px;margin:40px auto;padding:0 24px;color:#17212e;'
        'background:#f5f6f8}article{background:white;padding:24px;margin:24px 0;border-radius:8px}'
        'img{display:block;width:100%;margin-top:16px}p{line-height:1.55;color:#536071}</style>'
        '<h1>AFA · Thesis figure selection</h1><p>Curated temporal sequences and spatial–language correspondence. '
        'Exact frame and slot choices are saved; no object labels or slot matching are imposed.</p>' + ''.join(sections))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory containing selection.json and analysis/arrays")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    output = args.output or args.root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    study = Study(args.root, output, args.dpi)
    for d in study.arrays.values():
        for view in (0, 1):
            p = d[f"language_attention_{view}"].mean(0)
            if p.max() > .5:
                raise ValueError("Language display range would clip measured probabilities")
            np.testing.assert_allclose(p.sum(-1), 1, atol=1e-5)
    target_selection(study)
    temporal(study, "cube_demo", [5, 11, 15], 0, "main", "M02_cube_slot_evolution.",
             "Slot attention during cube placement",
             "Temporal evolution in a verified cube-placement demonstration (episode 49). The top row shows RGB; "
             "subsequent rows show the same learned queries S05, S11 and S15 at four selected interaction stages. "
             "Query IDs and display scales remain fixed. Maps may overlap or shift between scene regions; no "
             "instance tracking or per-frame query reassignment is performed. H is normalized spatial entropy. "
             "This is offline replay of a recorded demonstration, not a policy success measured in this analysis.")
    redundancy(study)
    temporal(study, "tool_demo", [6, 9, 13], 1, "main", "M04_tool_slot_evolution.",
             "Slot attention during tool use",
             "Alternative main-text temporal example from verified tool-use demonstration episode 149. The same "
             "wrist queries S06, S09 and S13 are shown while the visible scene changes from approach to tool/cube "
             "interaction and a late frame. In contrast to concentrated queries, S09 also reveals diffuse/background "
             "pooling. Full input padding is retained. These query IDs do not imply named object categories.")
    language_pair(study, "cube_demo", [5, 11, 15, 4], 0, [0, 1, 3],
                  "A01_cube_spatial_language.", "Cube placement: spatial and language attention")
    language_pair(study, "plush_failure", [3, 6, 13, 9], 1, [0, 1, 3],
                  "A02_failure_spatial_language.", "Wrong-object rollout: spatial and language attention")
    language_pair(study, "tool_demo", [3, 6, 13, 9], 1, [0, 1, 2],
                  "A03_tool_spatial_language.", "Tool use: spatial and language attention")
    full_language(study)
    atlas(study, "cube_detail", 0, 0, "A05_all_slots_cube_front.")
    atlas(study, "plush_failure", 1, 1, "A06_all_slots_failure_wrist.")
    temporal(study, "tool_failure", [5, 11, 15], 0, "appendix", "A07_tool_failure_evolution.",
             "Tool-task failure: fixed-query evolution",
             "Temporal spatial pooling for selected front queries in recorded tool-task failure episode 0. "
             "This sequence supplements the verified tool-use example. Timestamps and scene configurations "
             "differ across episodes; the figure is illustrative and does not isolate outcome as an experimental variable.")
    language_pair(study, "plush_demo", [3, 6, 13, 9], 1, [0, 1, 3],
                  "A08_demo_spatial_language.", "Verified demonstration: spatial and language attention")
    write_artifacts(study)
    print(f"Rendered {len(study.figures)} thesis figures: {output}", flush=True)


if __name__ == "__main__":
    main()
