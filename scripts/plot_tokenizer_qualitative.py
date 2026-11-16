#!/usr/bin/env python
"""Publication figures from tokenizer_qualitative.py arrays, with shared scales."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse
import numpy as np
from PIL import Image


COLORS = np.asarray(plt.get_cmap("tab20").colors)[[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 13]]
GROUPS = {"failure": "Failed policy rollout", "verified_demo": "Verified demonstration"}
TITLES = {"Grab the red cube and place on the gray block": "Cube placement",
          "Grab the plush and place in the box": "Plush placement",
          "Grab the peg and push the red cube": "Peg-assisted pushing"}


def unit_rows(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-10)


def probabilities(attention):
    p = attention.reshape(len(attention), -1).astype(float)
    return p / np.maximum(p.sum(-1, keepdims=True), 1e-12)


def routing(data, view):
    value = data.get(f"routing_{view}", data[f"attention_{view}"])
    value = value.reshape(data[f"attention_{view}"].shape)
    return value / np.maximum(value.sum(0, keepdims=True), 1e-12)


def enlarge(array, size, nearest=False):
    mode = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    if array.ndim == 2:
        return np.asarray(Image.fromarray(array.astype("float32")).resize((size, size), mode))
    return np.stack([enlarge(array[..., i], size, nearest) for i in range(array.shape[-1])], -1)


def image_axis(ax, image, title):
    ax.imshow(np.clip(image, 0, 1), extent=(-1, 1, 1, -1))
    ax.set(xlim=(-1, 1), ylim=(1, -1))
    ax.axis("off")
    if title:
        ax.set_title(title, loc="left", fontsize=10, pad=8, fontweight="medium")


def geometry(ax, data, view, labels=True, ellipses=True):
    p = probabilities(data[f"attention_{view}"])
    h, w = data[f"attention_{view}"].shape[-2:]
    yy, xx = np.meshgrid(np.linspace(-1 + 1/h, 1 - 1/h, h),
                         np.linspace(-1 + 1/w, 1 - 1/w, w), indexing="ij")
    grid = np.stack([xx.ravel(), yy.ravel()], -1)
    centers = data[f"keypoints_{view}"] * np.array([(w - 1) / w, (h - 1) / h])
    placed = []
    for k, center in enumerate(centers):
        if ellipses:
            delta = grid - center
            cov = (delta * p[k, :, None]).T @ delta
            values, vectors = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
            ellipse = Ellipse(center, 2 * np.sqrt(max(values[1], 0)),
                              2 * np.sqrt(max(values[0], 0)), angle=angle,
                              facecolor=COLORS[k], edgecolor=COLORS[k], alpha=0.16, lw=1)
            ax.add_patch(ellipse)
        ax.scatter(*center, s=32 if labels else 18, c=[COLORS[k]], edgecolors="white", linewidths=0.8, zorder=4)
        if labels:
            candidates = [np.clip(center + radius * np.array([np.cos(angle), np.sin(angle)]), -.91, .91)
                          for radius in (.08, .16, .25, .34, .44)
                          for angle in np.linspace(0, 2*np.pi, 12, endpoint=False)]
            label = next((point for point in candidates
                          if all(np.linalg.norm(point - other) > .115 for other in placed)), candidates[-1])
            placed.append(label)
            ax.annotate(str(k), center, xytext=label, fontsize=6.5, ha="center", va="center",
                        color="#172033", bbox=dict(boxstyle="round,pad=.15", fc="white", ec=COLORS[k], alpha=.95),
                        arrowprops=dict(arrowstyle="-", color=COLORS[k], lw=.6), zorder=5)


def save(fig, path, dpi):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=dpi, facecolor="white", bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


def fit_semantics(records, arrays, view):
    x = np.concatenate([arrays[r["id"]][f"semantics_{view}"] for r in records])
    mean = x.mean(0)
    _, _, vectors = np.linalg.svd(x - mean, full_matrices=False)
    basis = vectors[:3].T
    # Fix arbitrary PCA signs deterministically.
    for i in range(3):
        if basis[np.abs(basis[:, i]).argmax(), i] < 0:
            basis[:, i] *= -1
    scores = (x - mean) @ basis
    bounds = np.quantile(scores, [.02, .98], axis=0)
    return mean, basis, bounds


def semantic_colors(data, view, pca):
    mean, basis, bounds = pca
    scores = (data[f"semantics_{view}"] - mean) @ basis
    return np.clip((scores - bounds[0]) / np.maximum(bounds[1] - bounds[0], 1e-8), .08, .92)


def action_weights(data, view, k):
    # ACT memory: visual tokens (view-major), text tokens, proprioception.
    weights = data["action_attention"].mean(axis=(0, 1))
    return weights[view * k:(view + 1) * k]


def action_map(data, view):
    a = data[f"attention_{view}"]
    p = probabilities(a)
    weights = action_weights(data, view, len(a))
    if weights.sum() <= 0:
        return np.zeros(a.shape[-2:])
    return ((weights / weights.sum()) @ p).reshape(a.shape[-2:]) * p.shape[-1]


def overview(record, data, view, pca, output, dpi):
    rgb = data[f"rgb_{view}"]
    size = rgb.shape[0]
    a = data[f"attention_{view}"]
    assignment = routing(data, view)
    winners = assignment.argmax(0)
    # Confidence from the margin over uniform; ambiguous pixels remain RGB.
    confidence = (assignment.max(0) - 1 / len(a)) / (1 - 1 / len(a))
    clusters = enlarge(COLORS[winners], size, nearest=True)
    alpha = .65 * enlarge(confidence, size, nearest=True)[..., None]
    sem = np.einsum("khw,kc->hwc", assignment, semantic_colors(data, view, pca))
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 7.6))
    fig.subplots_adjust(top=.88, bottom=.17, wspace=.20, hspace=.19)
    camera = "Front" if view == 0 else "Wrist"
    fig.suptitle(f"{TITLES[record['task']]}  |  {GROUPS[record['group']]}", x=.125, ha="left", fontsize=14)
    fig.text(.125, .914, f"{camera} camera  ·  Episode {record['episode']}  ·  t = {record['frame'] / record['fps']:.2f} s",
             color="#667085", fontsize=9)
    image_axis(axes[0, 0], rgb, "a   Policy RGB input")
    image_axis(axes[0, 1], rgb, "b   Slot geometry · mean ± 1σ")
    geometry(axes[0, 1], data, view)
    image_axis(axes[0, 2], rgb * (1 - alpha) + clusters * alpha, "c   Soft slot assignment")
    image_axis(axes[1, 0], .38 * rgb + .62 * enlarge(sem, size), "d   Pooled semantics · shared PCA")
    geometry(axes[1, 0], data, view, labels=False, ellipses=False)
    image_axis(axes[1, 1], rgb, "e   Slot competition entropy")
    entropy = -(assignment * np.log(np.maximum(assignment, 1e-12))).sum(0) / np.log(len(a))
    im = axes[1, 1].imshow(entropy, extent=(-1, 1, 1, -1), cmap="cividis", vmin=0, vmax=1, alpha=.72)
    fig.colorbar(im, cax=axes[1, 1].inset_axes([0, -.06, 1, .025]),
                 orientation="horizontal", ticks=[0, .5, 1]).set_label("H / log K", fontsize=7)
    image_axis(axes[1, 2], rgb, "f   Action-to-slot projection")
    amap = action_map(data, view)
    lift = np.maximum(np.log2(np.maximum(amap, 1)), 0)
    overlay = plt.get_cmap("magma")(np.clip(enlarge(lift, size) / 3, 0, 1))
    overlay[..., 3] = .8 * np.clip(enlarge(lift, size) / 3, 0, 1)
    axes[1, 2].imshow(overlay, extent=(-1, 1, 1, -1))
    sm = plt.cm.ScalarMappable(norm=Normalize(0, 3), cmap="magma")
    fig.colorbar(sm, cax=axes[1, 2].inset_axes([0, -.06, 1, .025]),
                 orientation="horizontal", ticks=[0, 1, 2, 3]).set_label("log₂ lift", fontsize=7)
    visual_mass = action_weights(data, view, len(a)).sum()
    fig.text(.125, .045,
             f"K = {len(a)} slots · PCA fitted jointly per camera · decoder mass to this view: {visual_mass:.1%}\n"
             "Attention and soft assignments are diagnostic views; colors do not denote semantic classes or causal importance.",
             fontsize=8, color="#667085", linespacing=1.6)
    save(fig, output / "overview" / f"{record['id']}_{camera.lower()}", dpi)


def atlas(record, data, view, output, dpi):
    a = data[f"attention_{view}"]
    p = probabilities(a)
    rgb = data[f"rgb_{view}"]
    size = rgb.shape[0]
    k = len(a)
    columns = 4
    fig, axes = plt.subplots(int(np.ceil(k / columns)), columns, figsize=(9.8, 10.4))
    fig.subplots_adjust(top=.88, bottom=.08, wspace=.06, hspace=.17)
    camera = "front" if view == 0 else "wrist"
    fig.suptitle(f"What each slot pools  |  {TITLES[record['task']]}", x=.125, ha="left", fontsize=14)
    fig.text(.125, .92, f"{GROUPS[record['group']]} · {camera} · ep {record['episode']} · {record['frame']/record['fps']:.2f} s",
             color="#667085", fontsize=9)
    entropies = -(p * np.log(np.maximum(p, 1e-12))).sum(-1) / np.log(p.shape[-1])
    for slot, ax in enumerate(axes.flat):
        if slot >= k:
            ax.axis("off")
            continue
        lift = np.maximum(np.log2(np.maximum(p[slot].reshape(a.shape[-2:]) * p.shape[-1], 1)), 0)
        heat = np.clip(enlarge(lift, size) / 3, 0, 1)
        rgba = plt.get_cmap("magma")(heat)
        rgba[..., 3] = heat * .75
        image_axis(ax, rgb, f"S{slot:02d}   H / log N = {entropies[slot]:.2f}")
        ax.imshow(rgba, extent=(-1, 1, 1, -1))
        h, w = a.shape[-2:]
        center = data[f"keypoints_{view}"][slot] * np.array([(w-1)/w, (h-1)/h])
        ax.scatter(*center, s=25, c=[COLORS[slot]], edgecolors="white", lw=.8)
    cax = fig.add_axes([.36, .048, .34, .012])
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 3), cmap="magma"), cax=cax,
                 orientation="horizontal", ticks=[0, 1, 2, 3], label="log₂ attention lift over uniform · shared scale")
    save(fig, output / "slots" / f"{record['id']}_{camera}", dpi)


def diagnostics(record, data, view, output, dpi):
    p = probabilities(data[f"attention_{view}"])
    centered = unit_rows(p - 1 / p.shape[-1])
    sem = unit_rows(data[f"semantics_{view}"])
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    fig.subplots_adjust(top=.75, bottom=.22, wspace=.36)
    camera = "front" if view == 0 else "wrist"
    fig.suptitle(f"Slot structure  |  {TITLES[record['task']]}  |  {camera}", x=.125, ha="left", fontsize=13)
    for ax, matrix, title in [(axes[0], centered @ centered.T, "Attention similarity · uniform floor removed"),
                               (axes[1], sem @ sem.T, "Pooled semantic cosine similarity")]:
        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r", interpolation="nearest")
        ax.set_title(title, fontsize=9, loc="left")
        ax.set(xlabel="Slot", ylabel="Slot", xticks=[0, 5, 10, 15], yticks=[0, 5, 10, 15])
        fig.colorbar(im, ax=ax, fraction=.04, pad=.025, ticks=[-1, 0, 1])
    lang = data.get(f"language_attention_{view}")
    if lang is not None:
        im = axes[2].imshow(lang.mean(0), cmap="cividis", vmin=0, vmax=1, aspect="auto")
        axes[2].set_title("Early-BERT · query-to-word attention", loc="left", fontsize=9)
        axes[2].set_xticks(np.arange(len(data["words"])), data["words"], rotation=65, ha="right", fontsize=7)
        axes[2].set(ylabel="Slot", yticks=[0, 5, 10, 15])
        fig.colorbar(im, ax=axes[2], fraction=.04, pad=.025, ticks=[0, .5, 1])
    else:
        axes[2].axis("off")
    fig.text(.125, .02, "High off-diagonal attention similarity suggests redundant spatial pooling; semantic similarity is a separate property.",
             fontsize=8, color="#667085")
    save(fig, output / "diagnostics" / f"{record['id']}_{camera}", dpi)


def timeline(records, arrays, view, output, dpi):
    first = records[0]
    camera = "front" if view == 0 else "wrist"
    fig, axes = plt.subplots(2, len(records), figsize=(13, 5.6), squeeze=False)
    fig.subplots_adjust(top=.81, bottom=.09, hspace=.12, wspace=.035)
    fig.suptitle(f"{TITLES[first['task']]}  |  {GROUPS[first['group']]}  |  {camera}", x=.125, ha="left", fontsize=14)
    fig.text(.125, .86, f"Episode {first['episode']} · fixed relative times · identical slot colors within this camera", fontsize=9, color="#667085")
    for col, record in enumerate(records):
        data = arrays[record["id"]]
        rgb = data[f"rgb_{view}"]
        image_axis(axes[0, col], rgb, f"{record['frame']/record['fps']:.2f} s  ·  {record['fraction']:.0%}")
        geometry(axes[0, col], data, view, labels=False, ellipses=False)
        image_axis(axes[1, col], rgb, "")
        heat = np.clip(enlarge(np.maximum(np.log2(np.maximum(action_map(data, view), 1)), 0), len(rgb)) / 3, 0, 1)
        overlay = plt.get_cmap("magma")(heat)
        overlay[..., 3] = .8 * heat
        axes[1, col].imshow(overlay, extent=(-1, 1, 1, -1))
    fig.text(.08, .64, "Slot means", rotation=90, fontsize=9, va="center", color="#667085")
    fig.text(.08, .29, "Action projection", rotation=90, fontsize=9, va="center", color="#667085")
    fig.text(.125, .025, "Times are normalized within each episode, not aligned manipulation phases. Action projection uses log₂ lift, clipped to [0, 3].",
             fontsize=8, color="#667085")
    save(fig, output / "timelines" / f"task{first['task_id']}_{first['group']}_ep{first['episode']:03d}_{camera}", dpi)


def metric_row(record, data, view):
    a = data[f"attention_{view}"]
    p = probabilities(a)
    centered = unit_rows(p - 1 / p.shape[-1])
    similarity = centered @ centered.T
    k = len(a)
    assign = routing(data, view)
    return dict(id=record["id"], task=record["task"], group=record["group"], episode=record["episode"],
                frame=record["frame"], camera="front" if view == 0 else "wrist",
                spatial_entropy=float(-(p * np.log(np.maximum(p, 1e-12))).sum(-1).mean() / np.log(p.shape[-1])),
                centered_attention_cosine=float(similarity[~np.eye(k, dtype=bool)].mean()),
                assignment_entropy=float(-(assign * np.log(np.maximum(assign, 1e-12))).sum(0).mean() / np.log(k)),
                decoder_view_mass=float(action_weights(data, view, k).sum()),
                max_video_sync_error_s=float(np.abs(data["video_pts"][:, 0] - data["video_pts"][:, 1]).max()))


def comparison(records, arrays, view, pca, output, dpi):
    """Both failures and both demonstrations; no favorable-case selection."""
    records = sorted(records, key=lambda r: (r["group"], r["episode"]))
    camera = "front" if view == 0 else "wrist"
    fig, axes = plt.subplots(len(records), 3, figsize=(9.5, 12.5), squeeze=False)
    fig.subplots_adjust(top=.90, bottom=.07, left=.18, right=.96, wspace=.04, hspace=.08)
    fig.suptitle(f"{TITLES[records[0]['task']]}  |  {camera} camera", x=.18, ha="left", fontsize=14)
    fig.text(.18, .935, "Failed rollouts and verified demonstrations · episode midpoint", color="#667085", fontsize=9)
    for row, record in enumerate(records):
        data = arrays[record["id"]]
        rgb = data[f"rgb_{view}"]
        size = rgb.shape[0]
        assignment = routing(data, view)
        colors = semantic_colors(data, view, pca)
        sem = np.einsum("khw,kc->hwc", assignment, colors)
        image_axis(axes[row, 0], rgb, "Slot geometry" if row == 0 else "")
        geometry(axes[row, 0], data, view, labels=False)
        image_axis(axes[row, 1], .38 * rgb + .62 * enlarge(sem, size), "Pooled semantics" if row == 0 else "")
        image_axis(axes[row, 2], rgb, "Action-to-slot projection" if row == 0 else "")
        heat = np.clip(enlarge(np.maximum(np.log2(np.maximum(action_map(data, view), 1)), 0), size) / 3, 0, 1)
        overlay = plt.get_cmap("magma")(heat)
        overlay[..., 3] = .8 * heat
        axes[row, 2].imshow(overlay, extent=(-1, 1, 1, -1))
        axes[row, 0].text(-.10, .5,
                          f"{'Failure' if record['group'] == 'failure' else 'Verified demo'}\n"
                          f"ep {record['episode']} · {record['frame']/record['fps']:.2f} s",
                          transform=axes[row, 0].transAxes, ha="right", va="center", fontsize=9,
                          color="#a33e46" if record["group"] == "failure" else "#267565")
    fig.text(.18, .025, "Shared PCA per camera and fixed attention scale. Midpoints are not aligned task phases.\n"
             "Demonstrations provide correct recorded behavior; they do not measure closed-loop policy success.",
             color="#667085", fontsize=8, linespacing=1.5)
    save(fig, output / "comparisons" / f"task{records[0]['task_id']}_{camera}", dpi)


def embedding_space(records, arrays, view, output, dpi):
    from matplotlib.lines import Line2D

    x = np.concatenate([arrays[r["id"]][f"tokens_{view}"] for r in records])
    mean = x.mean(0)
    _, values, vectors = np.linalg.svd(x - mean, full_matrices=False)
    basis = vectors[:2].T
    variance = values[:2]**2 / np.sum(values**2)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), sharex=True, sharey=True)
    fig.subplots_adjust(top=.78, bottom=.27, wspace=.17)
    camera = "front" if view == 0 else "wrist"
    fig.suptitle(f"Fused tokenizer output space  |  {camera}", x=.125, ha="left", fontsize=14)
    fig.text(.125, .84, "Geometry + pooled semantics, before the AFA transformer · shared PCA across all observations", fontsize=9, color="#667085")
    for ax, task in zip(axes, dict.fromkeys(r["task"] for r in records)):
        for r in records:
            if r["task"] != task:
                continue
            scores = (arrays[r["id"]][f"tokens_{view}"] - mean) @ basis
            if r["group"] == "failure":
                ax.scatter(*scores.T, s=17, marker="o", facecolors="none", edgecolors=COLORS, alpha=.55, linewidths=.7)
            else:
                ax.scatter(*scores.T, s=17, marker="x", c=COLORS, alpha=.55, linewidths=.7)
        ax.set_title(TITLES[task], loc="left", fontsize=10)
        ax.set_xlabel(f"PC1 ({variance[0]:.1%})")
        ax.grid(alpha=.15)
    axes[0].set_ylabel(f"PC2 ({variance[1]:.1%})")
    handles = [Line2D([], [], color=COLORS[k], marker="o", ls="", label=f"S{k:02d}", markersize=5) for k in range(16)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .075), ncol=8, frameon=False, fontsize=8)
    fig.text(.125, .025, "○ Failed rollout   × Verified demonstration · separation can reflect learned slot identity and does not prove object-instance clustering.",
             fontsize=8, color="#667085")
    save(fig, output / "embeddings" / f"fused_tokens_{camera}", dpi)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--sections", nargs="+", default=["overview", "slots", "diagnostics", "timelines", "comparisons", "embeddings"],
                        choices=["overview", "slots", "diagnostics", "timelines", "comparisons", "embeddings"])
    args = parser.parse_args()
    output = args.output or args.analysis / "figures"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    manifest = json.loads((args.analysis / "manifest.json").read_text())
    records = manifest["records"]
    arrays = {r["id"]: dict(np.load(args.analysis / "arrays" / f"{r['id']}.npz", allow_pickle=False)) for r in records}
    output.mkdir(parents=True, exist_ok=True)
    pcas = [fit_semantics(records, arrays, view) for view in range(2)]
    np.savez(output / "semantic_pca.npz", **{f"{key}_{view}": value for view, pca in enumerate(pcas)
                                          for key, value in zip(("mean", "basis", "bounds"), pca)})
    metrics = [metric_row(r, arrays[r["id"]], v) for r in records for v in range(2)]
    with (output / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    grouped = {}
    for r in records:
        grouped.setdefault((r["task"], r["group"], r["episode"]), []).append(r)
    selected = []
    for group in grouped.values():
        group.sort(key=lambda r: r["fraction"])
        middle = min(group, key=lambda r: abs(r["fraction"] - .5))
        selected.append(middle)
        for view in range(2):
            if "overview" in args.sections:
                overview(middle, arrays[middle["id"]], view, pcas[view], output, args.dpi)
            if "slots" in args.sections:
                atlas(middle, arrays[middle["id"]], view, output, args.dpi)
            if "diagnostics" in args.sections:
                diagnostics(middle, arrays[middle["id"]], view, output, args.dpi)
            if "timelines" in args.sections:
                timeline(group, arrays, view, output, args.dpi)
        print(f"Rendered {middle['id']}", flush=True)
    if "comparisons" in args.sections:
        for task in dict.fromkeys(r["task"] for r in selected):
            for view in range(2):
                comparison([r for r in selected if r["task"] == task], arrays, view, pcas[view], output, args.dpi)
    if "embeddings" in args.sections:
        for view in range(2):
            embedding_space(records, arrays, view, output, args.dpi)
    # A self-contained local index permits inspecting every figure without rerunning inference.
    sections = []
    for directory in ("comparisons", "overview", "slots", "diagnostics", "timelines", "embeddings"):
        cards = "".join(f'<a href="{path.relative_to(output)}"><img loading="lazy" src="{path.relative_to(output)}"><span>{path.stem}</span></a>'
                        for path in sorted((output / directory).glob("*.png")))
        sections.append(f"<h2>{directory.title()}</h2><div class=grid>{cards}</div>")
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>AFA tokenizer qualitative study</title>'
        '<style>body{font:16px system-ui;margin:48px;background:#f5f6f8;color:#172033;max-width:1600px}'
        '.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}a{background:white;padding:14px;'
        'text-decoration:none;color:inherit;border-radius:10px}img{width:100%}span{display:block;font-size:11px;overflow-wrap:anywhere}</style>'
        '<h1>AFA · Offline tokenizer study</h1><p>Six failed policy rollouts and six verified demonstrations. '
        'The demonstrations are replayed observations, not measured policy successes. Figures show genuine model diagnostics; '
        'attention does not establish causal importance. Each PNG has a matching PDF.</p>' + "".join(sections))
    print(f"Gallery: {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
