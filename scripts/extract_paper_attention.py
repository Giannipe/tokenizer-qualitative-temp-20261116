#!/usr/bin/env python
"""Paired DINO CLS / AFA spatial diagnostics, preserving the real policy forward."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq

from tokenizer_qualitative import decode_frame, sha256


def selected_records(source, curated):
    source_data = json.loads(source.read_text())
    records = {}
    for r in source_data["records"]:
        if r["group"] == "verified_demo":
            records[r["id"]] = dict(r, selection_origin="relative_time")
    sequences = {}
    if curated:
        extra = json.loads(curated.read_text())
        for r in extra["records"]:
            if r["group"] == "verified_demo" and r["id"] not in records:
                records[r["id"]] = dict(r, selection_origin="RGB_event_inspection")
        for seq in extra["sequences"]:
            if all(ident in records for ident in seq["record_ids"]):
                sequences[seq["name"]] = seq
    return source_data, list(records.values()), sequences


def raw_batch(record, config, tables):
    import torch

    root = Path(record["root"])
    info = json.loads((root / "meta/info.json").read_text())
    ep = record["episode_metadata"]
    path = root / info["data_path"].format(chunk_index=ep["data/chunk_index"], file_index=ep["data/file_index"])
    if path not in tables:
        tables[path] = pq.read_table(path).to_pandas()
    rows = tables[path]
    row = rows[(rows.episode_index == record["episode"]) & (rows.frame_index == record["frame"])]
    if len(row) != 1:
        raise ValueError(f"Observation not unique: {record['id']}")
    row = row.iloc[0]
    raw = {"task": [record["task"]], "observation.state": torch.tensor(
        np.asarray(row["observation.state"], dtype=np.float32)).unsqueeze(0)}
    pts = []
    for key in config.camera_keys:
        path = root / info["video_path"].format(video_key=key,
            chunk_index=ep[f"videos/{key}/chunk_index"], file_index=ep[f"videos/{key}/file_index"])
        requested = ep[f"videos/{key}/from_timestamp"] + float(row.timestamp)
        rgb, actual = decode_frame(path, requested, record["fps"])
        raw[key] = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()[None] / 255
        pts.append([requested, actual])
    return raw, np.asarray(pts)


def extract(args):
    import torch
    import importlib
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from transformers.models.dinov3_vit.modeling_dinov3_vit import apply_rotary_pos_emb, eager_attention_forward

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    original, records, sequences = selected_records(args.source_manifest, args.curated_selection)
    policy_type = json.loads((args.checkpoint / "config.json").read_text())["type"]
    if policy_type not in {"turbovla", "turbovla_afa"}:
        raise ValueError("Supported checkpoints: turbovla, turbovla_afa")
    importlib.import_module(f"lerobot_policy_{policy_type}")
    output = args.output
    (output / "arrays").mkdir(parents=True, exist_ok=True)
    manifest = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha256(args.checkpoint / "model.safetensors"),
        policy_type=policy_type, device="cpu", precision="fp32", records=records, sequences=sequences,
        sources=original["sources"], source_checkpoint_sha256=original["checkpoint_sha256"],
        dino_definition="Last-layer CLS query to patch keys after RoPE; mean over heads; exclude CLS/register keys then spatially normalize",
        config=json.loads((args.checkpoint / "config.json").read_text()))
    path = output / "manifest.json"
    if path.exists():
        previous = json.loads(path.read_text())
        for key in ("checkpoint_sha256", "records", "policy_type"):
            if previous[key] != manifest[key]:
                raise ValueError(f"Cannot reuse output with different {key}")
    pending = [r for r in records if not (output / "arrays" / f"{r['id']}.npz").exists()]
    if not pending:
        print(f"All {len(records)} observations already extracted.")
        return
    config = PreTrainedConfig.from_pretrained(args.checkpoint, cli_overrides=["--device=cpu",
        "--inference_dtype=null", "--use_amp=false", "--vision.compute_precision=fp32", "--interaction.compute_precision=fp32"])
    policy = get_policy_class(policy_type).from_pretrained(args.checkpoint, config=config,
        local_files_only=True, strict=True).eval()
    pre, post = make_pre_post_processors(config, pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}})
    core = policy.model
    attention_modules = [m for m in core.vision_encoder.backbone.modules() if m.__class__.__name__ == "DINOv3ViTAttention"]
    last = attention_modules[-1]
    prefix = core.vision_encoder.prefix_tokens
    state = {"capture": True, "verified_eager": False}
    current = {}

    def dino_pre(module, inputs, kwargs):
        if not state["capture"]:
            return
        x = inputs[0] if inputs else kwargs["hidden_states"]
        b, n, _ = x.shape
        q = module.q_proj(x).view(b, n, module.num_heads, module.head_dim).transpose(1, 2)
        k = module.k_proj(x).view(b, n, module.num_heads, module.head_dim).transpose(1, 2)
        cos, sin = kwargs["position_embeddings"]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        scores = (q[:, :, :1] @ k.transpose(-2, -1)) * module.scaling
        mask = kwargs.get("attention_mask")
        if mask is not None:
            scores = scores + mask[..., :1, :]
        weights = scores.softmax(-1)
        if not state["verified_eager"]:
            v = module.v_proj(x).view(b, n, module.num_heads, module.head_dim).transpose(1, 2)
            _, reference = eager_attention_forward(module, q, k, v, mask, scaling=module.scaling, dropout=0)
            error = float((weights - reference[:, :, :1]).abs().max())
            if error > 1e-6:
                raise ValueError(f"CLS replay differs from eager attention: {error}")
            manifest["cls_replay_vs_eager_max_abs_error"] = error
            state["verified_eager"] = True
        current.setdefault("cls", []).append(weights[:, :, 0].detach().cpu())

    def vision_hook(module, inputs, output):
        if state["capture"]:
            current["features"] = output.detach().cpu()

    def act_hook(module, inputs, kwargs, output):
        if state["capture"]:
            _, weights = module.forward(*inputs, **dict(kwargs, need_weights=True, average_attn_weights=False))
            current["act"] = weights.detach().cpu()[0].mean((0, 1)).numpy()

    def pool_hook(view):
        def hook(module, inputs, output):
            if not state["capture"]:
                return
            for key, value in (("keypoints", output[0]), ("tokens", output[1]),
                ("attention", module.latest_attention), ("routing", module.latest_query_softmax),
                ("semantics", module.latest_semantic_embeddings)):
                current[f"{key}_{view}"] = value[0].detach().cpu().numpy()
        return hook

    handles = [last.register_forward_pre_hook(dino_pre, with_kwargs=True),
        core.vision_encoder.register_forward_hook(vision_hook),
        core.action_head.decoder.decoder.layers[-1].multihead_attn.register_forward_hook(act_hook, with_kwargs=True)]
    if policy_type == "turbovla_afa":
        for view, pool in enumerate(core.afa_tokenizers):
            pool.record_inference_diagnostics = True
            handles.append(pool.register_forward_hook(pool_hook(view)))
    tables = {}
    try:
        with torch.inference_mode():
            for idx, record in enumerate(records):
                destination = output / "arrays" / f"{record['id']}.npz"
                if destination.exists():
                    continue
                started = time.monotonic()
                raw, pts = raw_batch(record, config, tables)
                batch = pre(raw)
                current.clear()
                state["capture"] = True
                prediction = policy.predict_action_chunk(batch)
                captured = current.copy()
                state["capture"] = False
                baseline = policy.predict_action_chunk(batch)
                error = float((baseline - prediction).abs().max())
                if error > 1e-6:
                    raise ValueError(f"Diagnostic hooks changed prediction for {record['id']}: {error}")
                arrays = dict(video_pts=pts, predicted_actions=post(prediction).cpu().numpy()[0],
                              hook_prediction_max_abs_error=np.asarray(error))
                rows = torch.cat(captured["cls"], 0)
                if rows.shape[0] != 2:
                    raise ValueError(f"Expected two camera rows, got {rows.shape}")
                for view, key in enumerate(config.camera_keys):
                    mean = torch.tensor(config.image_mean).view(1, 3, 1, 1)
                    std = torch.tensor(config.image_std).view(1, 3, 1, 1)
                    arrays[f"rgb_{view}"] = ((batch[key].cpu()*std+mean)[0].permute(1, 2, 0).clamp(0, 1)).numpy()
                    all_weights = rows[view].numpy()
                    patch_weights = all_weights[:, prefix:]
                    avg = patch_weights.mean(0)
                    arrays[f"dino_cls_{view}"] = (avg / avg.sum()).reshape(16, 16)
                    arrays[f"dino_cls_head_weights_{view}"] = all_weights
                    arrays[f"dino_cls_patch_mass_{view}"] = patch_weights.sum(-1)
                    if record["selection_origin"] == "relative_time" and abs(record["fraction"]-.5) < 1e-6:
                        arrays[f"dino_features_{view}"] = captured["features"][0, view].numpy()
                    slots = 16 if policy_type == "turbovla_afa" else 256
                    act = captured["act"][view*slots:(view+1)*slots]
                    arrays[f"act_view_mass_{view}"] = np.asarray(act.sum())
                    if policy_type == "turbovla_afa":
                        for name in ("attention", "routing", "keypoints", "tokens", "semantics"):
                            arrays[f"{name}_{view}"] = captured[f"{name}_{view}"]
                        p = captured[f"attention_{view}"].reshape(slots, -1)
                        p = p / p.sum(-1, keepdims=True)
                        projected = (act/act.sum()) @ p
                    else:
                        projected = act/act.sum()
                    arrays[f"act_map_{view}"] = projected.reshape(16, 16)
                for key, value in arrays.items():
                    if np.asarray(value).dtype.kind in "fc" and not np.isfinite(value).all():
                        raise ValueError(f"Nonfinite {key}")
                temporary = destination.with_suffix(".npz.tmp")
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, **arrays)
                temporary.replace(destination)
                manifest["completed"] = len(list((output/"arrays").glob("*.npz")))
                manifest["dino_heads"] = last.num_heads
                manifest["dino_prefix_tokens"] = prefix
                manifest["hook_prediction_max_abs_error"] = max(error, manifest.get("hook_prediction_max_abs_error", 0))
                path.write_text(json.dumps(manifest, indent=2))
                print(f"[{idx+1}/{len(records)}] {policy_type} {record['id']} {time.monotonic()-started:.1f}s", flush=True)
    finally:
        for handle in handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--curated-selection", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    extract(parser.parse_args())


if __name__ == "__main__":
    main()
