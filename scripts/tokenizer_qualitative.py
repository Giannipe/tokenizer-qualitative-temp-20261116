#!/usr/bin/env python
"""Extract genuine AFA diagnostics offline; render separately without model inference.

Supports local LeRobot v3 video datasets. The manifest pins observations, task
strings, dataset revisions and checkpoint hash. No training or robot is involved.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq


def read_episodes(root):
    paths = sorted((root / "meta/episodes").rglob("*.parquet"))
    if not paths:
        raise ValueError(f"No LeRobot v3 episode metadata in {root}")
    return sorted([row for path in paths for row in pq.read_table(path).to_pylist()],
                  key=lambda row: row["episode_index"])


def decode_frame(path, timestamp, fps):
    """Nearest video PTS, explicitly checking synchronization to within one frame."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(max(0, int(timestamp / stream.time_base)), stream=stream, backward=True)
        best = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            pts = float(frame.pts * stream.time_base)
            if best is None or abs(pts - timestamp) < abs(best[0] - timestamp):
                best = (pts, frame.to_ndarray(format="rgb24"))
            if pts >= timestamp:
                break
        if best is None or abs(best[0] - timestamp) > 1.01 / fps:
            raise ValueError(f"Video synchronization failed: {path}, requested {timestamp}, got {best}")
        return best[1], best[0]


def build_manifest(args):
    success_root = args.success_root.resolve()
    successful = read_episodes(success_root)
    sources = []
    for root in sorted(args.failure_root.glob("rollout_tokenizer_attention_task*")):
        episodes = read_episodes(root)
        if len(episodes) != 2:
            raise ValueError(f"Expected two failure episodes: {root}")
        sources.append((root.resolve(), "failure", episodes))
    if len(sources) != 3:
        raise ValueError("Expected exactly the three rollout_tokenizer_attention_task datasets")
    tasks = list(dict.fromkeys(task for _, _, episodes in sources for ep in episodes for task in ep["tasks"]))
    for task in tasks:
        candidates = [ep for ep in successful if task in ep["tasks"]]
        if len(candidates) < 2:
            raise ValueError(f"Fewer than two verified demonstrations for {task}")
        selected = [candidates[0], candidates[-1]]
        sources.append((success_root, "verified_demo", selected))
    records = []
    for root, group, episodes in sources:
        info = json.loads((root / "meta/info.json").read_text())
        for ep in episodes:
            task = ep["tasks"][0]
            task_id = tasks.index(task) + 1
            for fraction in args.fractions:
                frame = round(fraction * (ep["length"] - 1))
                ident = f"task{task_id}_{group}_ep{ep['episode_index']:03d}_f{frame:05d}"
                records.append(dict(id=ident, root=str(root), group=group, task=task,
                                    task_id=task_id, episode=int(ep["episode_index"]),
                                    frame=frame, fraction=fraction, fps=info["fps"],
                                    episode_metadata=ep))
    # Discard bulky training statistics from the durable observation manifest.
    for record in records:
        record["episode_metadata"] = {key: value for key, value in record["episode_metadata"].items()
                                      if not key.startswith("stats/")}
    return records


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(args):
    import torch
    import lerobot_policy_turbovla_afa  # noqa: F401, register policy and processors
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu for offline extraction")
    records = build_manifest(args)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    manifest = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint / "model.safetensors"),
                    config=json.loads((checkpoint / "config.json").read_text()),
                    selection="Both failure episodes; first and last verified episode per task; fixed relative times",
                    records=records, device=args.device, torch_version=torch.__version__)
    manifest["sources"] = []
    for root in dict.fromkeys(record["root"] for record in records):
        root = Path(root)
        metadata = root / ".cache/huggingface/download/meta/info.json.metadata"
        revision = metadata.read_text().splitlines()[0] if metadata.exists() else None
        repo_id = "Giannipe/Multi-task-verified" if root == args.success_root.resolve() else f"Giannipe/{root.name}"
        manifest["sources"].append(dict(root=str(root), repo_id=repo_id, revision=revision,
                                        info_sha256=sha256(root / "meta/info.json")))
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in ("checkpoint_sha256", "records", "config", "device"):
            if previous[key] != manifest[key]:
                raise ValueError(f"Existing output has different {key}; use a new output directory")
        if "hook_prediction_max_abs_error" in previous:
            manifest["hook_prediction_max_abs_error"] = previous["hook_prediction_max_abs_error"]
    manifest_path.write_text(json.dumps(manifest, indent=2))
    if all((args.output / "arrays" / f"{record['id']}.npz").exists() for record in records):
        print(f"All {len(records)} observations already extracted; manifest verified.", flush=True)
        return
    config = PreTrainedConfig.from_pretrained(checkpoint, cli_overrides=[
        f"--device={args.device}", "--inference_dtype=null", "--use_amp=false",
        "--vision.compute_precision=fp32", "--interaction.compute_precision=fp32",
    ])
    policy = get_policy_class("turbovla_afa").from_pretrained(
        checkpoint, config=config, local_files_only=True, strict=True).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        config, pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": args.device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}})
    core = policy.model
    for module in core.modules():
        module.record_inference_diagnostics = True
    captured = {}

    def pool_hook(view):
        def hook(module, inputs, output):
            keypoints, tokens = output[:2]
            for name, value in (("keypoints", keypoints), ("tokens", tokens),
                                ("attention", module.latest_attention),
                                ("semantics", module.latest_semantic_embeddings),
                                ("routing", module.latest_query_softmax)):
                if value is not None:
                    captured[f"{name}_{view}"] = value.detach().float().cpu().numpy()
        return hook

    handles = [pool.register_forward_hook(pool_hook(i)) for i, pool in enumerate(core.afa_tokenizers)]

    def attention_hook(name):
        def hook(module, inputs, kwargs, output):
            # Replay only this small attention operation for diagnostics. Calling
            # forward directly avoids hooks and never replaces the real output.
            replay = dict(kwargs, need_weights=True, average_attn_weights=False)
            _, weights = module.forward(*inputs, **replay)
            captured[name] = weights.detach().float().cpu().numpy()
        return hook

    handles.append(core.action_head.decoder.decoder.layers[-1].multihead_attn.register_forward_hook(
        attention_hook("action_attention"), with_kwargs=True))
    for i, pool in enumerate(core.afa_tokenizers):
        if hasattr(pool, "task_query_cross_attention"):
            handles.append(pool.task_query_cross_attention.cross_attention.register_forward_hook(
                attention_hook(f"language_attention_{i}"), with_kwargs=True))
    frames_dir = args.output / "arrays"
    frames_dir.mkdir(exist_ok=True)
    data_cache = {}
    checked = False
    try:
        with torch.inference_mode():
            for index, record in enumerate(records):
                destination = frames_dir / f"{record['id']}.npz"
                if destination.exists():
                    continue
                start = time.monotonic()
                root = Path(record["root"])
                info = json.loads((root / "meta/info.json").read_text())
                ep = record["episode_metadata"]
                data_path = root / info["data_path"].format(
                    chunk_index=ep["data/chunk_index"], file_index=ep["data/file_index"])
                if data_path not in data_cache:
                    data_cache[data_path] = pq.read_table(data_path).to_pandas()
                table = data_cache[data_path]
                rows = table[(table.episode_index == record["episode"]) & (table.frame_index == record["frame"])]
                if len(rows) != 1:
                    raise ValueError(f"Ambiguous observation: {record['id']}")
                row = rows.iloc[0]
                raw = {"task": [record["task"]], "observation.state": torch.tensor(
                    np.asarray(row["observation.state"], dtype=np.float32)).unsqueeze(0)}
                video_pts = []
                for key in config.camera_keys:
                    video_path = root / info["video_path"].format(
                        video_key=key, chunk_index=ep[f"videos/{key}/chunk_index"],
                        file_index=ep[f"videos/{key}/file_index"])
                    timestamp = ep[f"videos/{key}/from_timestamp"] + float(row.timestamp)
                    rgb, pts = decode_frame(video_path, timestamp, record["fps"])
                    video_pts.append([timestamp, pts])
                    raw[key] = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255
                batch = preprocessor(raw)
                captured.clear()
                prediction = policy.predict_action_chunk(batch)
                if not torch.isfinite(prediction).all():
                    raise ValueError("Nonfinite policy prediction")
                # The diagnostic replay must preserve the actual policy output.
                if not checked:
                    saved = {key: value.copy() for key, value in captured.items()}
                    for handle in handles:
                        handle.remove()
                    baseline = policy.predict_action_chunk(batch)
                    error = float((baseline - prediction).abs().max())
                    if error > 1e-6:
                        raise ValueError(f"Diagnostic hooks changed predictions: {error}")
                    manifest["hook_prediction_max_abs_error"] = error
                    manifest_path.write_text(json.dumps(manifest, indent=2))
                    captured = saved
                    handles = [pool.register_forward_hook(pool_hook(i)) for i, pool in enumerate(core.afa_tokenizers)]
                    handles.append(core.action_head.decoder.decoder.layers[-1].multihead_attn.register_forward_hook(
                        attention_hook("action_attention"), with_kwargs=True))
                    for i, pool in enumerate(core.afa_tokenizers):
                        if hasattr(pool, "task_query_cross_attention"):
                            handles.append(pool.task_query_cross_attention.cross_attention.register_forward_hook(
                                attention_hook(f"language_attention_{i}"), with_kwargs=True))
                    checked = True
                arrays = {}
                mean = torch.tensor(config.image_mean).view(1, 3, 1, 1)
                std = torch.tensor(config.image_std).view(1, 3, 1, 1)
                for view, key in enumerate(config.camera_keys):
                    rgb = (batch[key].cpu() * std + mean)[0].permute(1, 2, 0).clamp(0, 1)
                    arrays[f"rgb_{view}"] = rgb.numpy()
                    pool_idx = view if len(core.afa_tokenizers) > 1 else 0
                    pool_row = 0 if len(core.afa_tokenizers) > 1 else view
                    for name in ("keypoints", "tokens", "attention", "semantics", "routing"):
                        source = captured.get(f"{name}_{pool_idx}")
                        if source is not None:
                            arrays[f"{name}_{view}"] = source[pool_row]
                    lang = captured.get(f"language_attention_{pool_idx}")
                    if lang is not None:
                        arrays[f"language_attention_{view}"] = lang[pool_row]
                arrays["action_attention"] = captured["action_attention"][0]
                arrays["predicted_actions"] = postprocessor(prediction).cpu().numpy()[0]
                arrays["recorded_action"] = np.asarray(row.action, dtype=np.float32)
                arrays["video_pts"] = np.asarray(video_pts)
                tokenizer = core.text_encoder.tokenizer
                arrays["words"] = np.asarray(tokenizer.convert_ids_to_tokens(
                    tokenizer(record["task"], truncation=True, max_length=config.text.max_length)["input_ids"]))
                temporary = destination.with_suffix(".npz.tmp")
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, **arrays)
                temporary.replace(destination)
                print(f"[{index + 1}/{len(records)}] {record['id']} {time.monotonic() - start:.1f}s", flush=True)
    finally:
        for handle in handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--failure-root", type=Path, required=True)
    parser.add_argument("--success-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    args = parser.parse_args()
    if any(not 0 <= value <= 1 for value in args.fractions):
        parser.error("Fractions must be in [0,1]")
    extract(args)


if __name__ == "__main__":
    main()
