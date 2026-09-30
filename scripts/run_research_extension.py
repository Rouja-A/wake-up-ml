#!/usr/bin/env python
"""Research extension: single entry point (Streaming Wake-Up Networks exploratory study).

    python scripts/run_research_extension.py --config configs/baseline.yaml --device cuda

Run from the project root. TRAIN-SUBJECT-ONLY: data comes exclusively from
`experiments.research_extension.data_ext` (built on `src.data.load_train_only`); the official test split
is never loaded, and nothing here evaluates on it. All outputs go under `results/research_extension/`
(checkpoints/, metrics/, figures/); frozen Phase-1/Phase-2 files are only ever READ (A/C Phase-1
checkpoints for per-subject analysis, split.json and architecture_selection.json for verification).

Runs (with the config's seeds [0,1,2]): Linear x3, D x3, RQ4 sensor ablation (Model-A architecture)
4 arms x3 = 12  ->  18 training runs. Then per-subject analysis, complexity table, Pareto plots.

Flags: --dry_run (print plan + verify guards, no data loaded, no training);
       --smoke (2 epochs, seed 0 only, outputs to results/research_extension/_smoke/ -- pipeline check only,
       numbers are NOT results).
"""
import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.research_extension.output_guard import (assert_no_existing_outputs,            # noqa: E402
                                                          assert_output_dir_safe)
from experiments.research_extension.pareto import ParetoPoint, pareto_frontier                  # noqa: E402
from experiments.research_extension.per_subject import per_subject_metrics, summarize_subject_variability  # noqa: E402
from experiments.research_extension.shape_calc_ext import (analyze_linear, analyze_model_d,     # noqa: E402
                                                            weight_storage_bytes)
from experiments.research_extension.streaming_state import total_streaming_state                 # noqa: E402
from src.architectures import ARCHITECTURES                                                       # noqa: E402
from src.shape_calc import analyze                                                                # noqa: E402
from src.utils import default_data_dir, load_config, save_json                                    # noqa: E402

SENSOR_ARMS = ("acc_gyro_full", "acc_only", "gyro_only", "body_acc_gyro")
EXPECTED_FROZEN = {"A": (3766, 274624), "B": (14390, 602496), "C": (8862, 415296)}   # params, MACs


def load_frozen_references() -> dict:
    """A/B/C mean/std validation macro-F1 READ from the frozen Phase-1 summary (never written), and
    params/MACs recomputed from the frozen specs; asserts they equal the documented values."""
    import json

    sel = json.loads((ROOT / "results" / "metrics" / "architecture_selection.json").read_text())["summary"]
    refs = {}
    for k, spec in ARCHITECTURES.items():
        rep = analyze(spec, k)
        if (rep.total_params, rep.total_macs) != EXPECTED_FROZEN[k]:
            raise RuntimeError(f"Frozen model {k} complexity changed: {(rep.total_params, rep.total_macs)}")
        refs[k] = {"params": rep.total_params, "macs": rep.total_macs, "receptive_field": rep.receptive_field,
                   "mean_val_macro_f1": sel[k]["mean_val_macro_f1"], "std_val_macro_f1": sel[k]["std_val_macro_f1"],
                   "causal": False, **weight_storage_bytes(rep)}
    return refs


def verify_split_matches_frozen(prepared) -> None:
    """The extension's train/val subjects must equal the frozen Phase-1 split (read-only comparison)."""
    import json

    frozen = json.loads((ROOT / "results" / "metrics" / "split.json").read_text())
    if list(prepared.train_subjects) != frozen["train_subjects"] or list(prepared.val_subjects) != frozen["val_subjects"]:
        raise RuntimeError("Extension split differs from frozen results/metrics/split.json -- aborting.")


def planned_run_names(seeds):
    names = [f"research_extension_Linear_seed{s}" for s in seeds] + [f"research_extension_D_seed{s}" for s in seeds]
    for arm in SENSOR_ARMS:
        names += [f"research_extension_ablation_{arm}_seed{s}" for s in seeds]
    return names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--overwrite_extension_outputs", action="store_true",
                    help="allow replacing files previously written under results/research_extension/ (never frozen files)")
    args = ap.parse_args()

    cfg = load_config(args.config if Path(args.config).is_absolute() else ROOT / args.config)
    seeds = [0] if args.smoke else list(cfg["training"]["phase1_seeds"])
    max_epochs = 2 if args.smoke else None
    base = ROOT / "results" / "research_extension" / ("_smoke" if args.smoke else "")
    ckpt_dir = assert_output_dir_safe(base / "checkpoints", ROOT)
    metrics_dir = assert_output_dir_safe(base / "metrics", ROOT)
    fig_dir = assert_output_dir_safe(base / "figures", ROOT)

    names = planned_run_names(seeds)
    planned = [ckpt_dir / f"{n}.pt" for n in names] + [metrics_dir / f"{n}_curves.json" for n in names] \
        + [metrics_dir / "research_extension_summary.json", metrics_dir / "research_extension_tables.md"]
    assert_no_existing_outputs(planned, args.overwrite_extension_outputs)

    print(f"Training runs planned: {len(names)}  (seeds {seeds}{', SMOKE: 2 epochs' if args.smoke else ''})")
    print(f"Outputs: {base}  (checkpoints/, metrics/, figures/). Official test split: never loaded.")
    if args.dry_run:
        for n in names:
            print("  -", n)
        print("Dry run only: no data loaded, nothing written.")
        return 0

    import torch
    from torch.utils.data import DataLoader

    from experiments.research_extension.architectures_ext import model_a_variant_spec
    from experiments.research_extension.data_ext import SENSOR_SUBSETS, prepare_channel_subset, prepare_full_six_channel
    from experiments.research_extension.model_ext import build_model_ext, build_sensor_ablation_model
    from experiments.research_extension.train_ext import fit_ext
    from src.dataset import HARDataset
    from src.evaluate import load_model_from_checkpoint, predict
    from src.train import get_device, load_checkpoint

    device = get_device(args.device)
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    for d_ in (ckpt_dir, metrics_dir, fig_dir):
        d_.mkdir(parents=True, exist_ok=True)

    data_dir = args.data_dir or cfg["data"]["data_dir"] or default_data_dir()
    n_val, split_seed = cfg["split"]["n_val_subjects"], cfg["split"]["seed"]
    tcfg = cfg["training"]
    bs = tcfg["batch_size"]
    frozen = load_frozen_references()

    def loaders(prep, seed):   # same construction as src.dataset.make_loaders (train shuffled with seeded generator)
        g = torch.Generator()
        g.manual_seed(seed)
        return (DataLoader(HARDataset(prep.x_train, prep.y_train), batch_size=bs, shuffle=True, generator=g, num_workers=0),
                DataLoader(HARDataset(prep.x_val, prep.y_val), batch_size=bs, shuffle=False, num_workers=0))

    counter = {"i": 0}

    def train_seeds(model_name, prep, tag, factory=None):
        results = []
        for seed in seeds:
            counter["i"] += 1
            run_name = f"research_extension_{tag}_seed{seed}"
            print(f"\n=== [run {counter['i']}/{len(names)}] {run_name} ===")
            tr, va = loaders(prep, seed)
            r = fit_ext(model_name, tr, va, prep.class_names, tcfg, seed=seed, run_name=run_name,
                        checkpoint_dir=ckpt_dir, metrics_dir=metrics_dir, device=device, max_epochs=max_epochs,
                        normalization_stats=prep.stats.to_dict(), channels=prep.channels, quiet=True,
                        model_factory=factory)
            print(f"  -> best_val_macro_f1={r.best_val_macro_f1:.4f} best_val_acc={r.best_val_acc:.4f} "
                  f"best_epoch={r.best_epoch} epochs_ran={r.epochs_ran} time={r.train_seconds:.1f}s")
            results.append(r)
        f1s = [r.best_val_macro_f1 for r in results]
        return {"mean_val_macro_f1": statistics.mean(f1s),
                "std_val_macro_f1": statistics.pstdev(f1s) if len(f1s) > 1 else 0.0,
                "min": min(f1s), "max": max(f1s), "n_seeds": len(f1s), "values": f1s,
                "best_epoch_per_seed": [r.best_epoch for r in results]}

    # ---- RQ1 / RQ2 (six-channel, train-only) ----
    prep = prepare_full_six_channel(data_dir, n_val, split_seed)
    verify_split_matches_frozen(prep)
    print(f"Split matches frozen Phase-1 split. train subjects={list(prep.train_subjects)} "
          f"val subjects={list(prep.val_subjects)}  ({len(prep.y_train)} train / {len(prep.y_val)} val windows)")
    linear = train_seeds("Linear", prep, "Linear")
    d = train_seeds("D", prep, "D")

    # ---- RQ4 sensor ablation (Model-A architecture, fixed in advance) ----
    ablation = {}
    for arm in SENSOR_ARMS:
        ch = SENSOR_SUBSETS[arm]
        prep_arm = prepare_channel_subset(data_dir, ch, n_val, split_seed)   # normalization refit on THIS subset
        verify_split_matches_frozen(prep_arm)
        rep = analyze(model_a_variant_spec(len(ch)), f"A_variant_{arm}", input_channels=len(ch))
        summ = train_seeds("A_variant", prep_arm, f"ablation_{arm}",
                           factory=lambda n=len(ch): build_sensor_ablation_model(n))
        ablation[arm] = {"channels": list(ch), "params": rep.total_params, "macs": rep.total_macs, **summ}

    # ---- per-subject validation analysis (validation subjects of the frozen split only) ----
    _, val_loader = loaders(prep, 0)

    def per_subject_from(model_getter, seeds_):
        per_seed = []
        for s in seeds_:
            yp, yt = predict(model_getter(s), val_loader, device)
            per_seed.append(per_subject_metrics(yt, yp, prep.subjects_val, prep.class_names))
        merged = []
        for i, row in enumerate(per_seed[0]):
            ents = [ps[i] for ps in per_seed]
            merged.append({"subject_id": row["subject_id"], "n_windows": row["n_windows"],
                           "accuracy": statistics.mean(e["accuracy"] for e in ents),
                           "macro_f1": statistics.mean(e["macro_f1"] for e in ents), "n_seeds_averaged": len(ents)})
        return {"per_subject": merged, "summary": summarize_subject_variability(merged)}

    def ext_getter(name):
        def g(seed):
            ck = load_checkpoint(ckpt_dir / f"research_extension_{name}_seed{seed}.pt", device)
            m = build_model_ext(name).to(device)
            m.load_state_dict(ck["state_dict"])
            return m.eval()
        return g

    def frozen_getter(name):   # READ-ONLY use of frozen Phase-1 checkpoints
        def g(seed):
            m, _ = load_model_from_checkpoint(ROOT / "results" / "checkpoints" / f"phase1_{name}_seed{seed}.pt", device)
            return m
        return g

    per_subject = {}
    frozen_seeds = list(cfg["training"]["phase1_seeds"])
    for name, getter, sds in (("A", frozen_getter("A"), frozen_seeds), ("C", frozen_getter("C"), frozen_seeds),
                              ("D", ext_getter("D"), seeds), ("Linear", ext_getter("Linear"), seeds)):
        per_subject[name] = per_subject_from(getter, sds)

    # ---- complexity + Pareto ----
    rd, rl = analyze_model_d(), analyze_linear()
    complexity = {"Linear": {"params": rl.total_params, "macs": rl.total_macs, "receptive_field": rl.receptive_field,
                             "causal": False, **weight_storage_bytes(rl)},
                  "D": {"params": rd.total_params, "macs": rd.total_macs, "receptive_field": rd.receptive_field,
                        "post_gap_coverage": rd.post_gap_coverage, "causal": True, **weight_storage_bytes(rd)},
                  **{k: {kk: vv for kk, vv in v.items() if kk not in ("mean_val_macro_f1", "std_val_macro_f1")}
                     for k, v in frozen.items()}}
    mean_f1 = {"Linear": linear["mean_val_macro_f1"], "D": d["mean_val_macro_f1"],
               **{k: v["mean_val_macro_f1"] for k, v in frozen.items()}}
    pareto = {axis: pareto_frontier([ParetoPoint(n, mean_f1[n], complexity[n][axis]) for n in mean_f1])
              for axis in ("macs", "params")}

    out = {"protocol": {"seeds": seeds, "smoke": args.smoke, "split_seed": split_seed,
                        "val_subjects": list(prep.val_subjects), "official_test_used": False},
           "linear_baseline": linear, "model_d": d, "sensor_ablation": ablation, "per_subject": per_subject,
           "frozen_reference_A_B_C": frozen, "complexity": complexity, "pareto": pareto,
           "streaming_state_d": total_streaming_state()}
    save_json(out, metrics_dir / "research_extension_summary.json")
    write_tables(out, mean_f1, metrics_dir / "research_extension_tables.md")
    plot_all(complexity, mean_f1, per_subject, fig_dir)
    print(f"\nWrote {metrics_dir / 'research_extension_summary.json'}, research_extension_tables.md, "
          f"figures in {fig_dir}")
    print("Done. Official test split was never loaded or evaluated.")
    return 0


def write_tables(out, mean_f1, path: Path) -> None:
    c, f = out["complexity"], out["frozen_reference_A_B_C"]
    std = {"Linear": out["linear_baseline"]["std_val_macro_f1"], "D": out["model_d"]["std_val_macro_f1"],
           **{k: v["std_val_macro_f1"] for k, v in f.items()}}
    L = ["# Research-extension results (auto-generated from measured runs; validation subjects only)", "",
         "## Models", "",
         "| Model | Mean val macro-F1 | Std (3 seeds) | Params | MAC/window | RF | Causal? |",
         "|---|---:|---:|---:|---:|---:|---|"]
    for n in ("Linear", "A", "B", "C", "D"):
        L.append(f"| {n} | {mean_f1[n]:.4f} | {std[n]:.4f} | {c[n]['params']:,} | {c[n]['macs']:,} | "
                 f"{c[n].get('receptive_field', '-')} | {'yes (backbone)' if c[n]['causal'] else 'no'} |")
    L += ["", "A/B/C rows are read from the frozen Phase-1 summary; Linear/D are newly trained in this extension.", "",
          "## Sensor ablation (Model-A architecture)", "",
          "| Arm | Channels | Params | MAC/window | Mean val macro-F1 | Std | Min | Max |",
          "|---|---|---:|---:|---:|---:|---:|---:|"]
    for arm, v in out["sensor_ablation"].items():
        L.append(f"| {arm} | {len(v['channels'])} | {v['params']:,} | {v['macs']:,} | {v['mean_val_macro_f1']:.4f} | "
                 f"{v['std_val_macro_f1']:.4f} | {v['min']:.4f} | {v['max']:.4f} |")
    L += ["", "## Per-validation-subject macro-F1 (mean over seeds)", ""]
    for name, r in out["per_subject"].items():
        s = r["summary"]
        L += [f"### {name}", "", "| Subject | Windows | Accuracy | Macro-F1 |", "|---:|---:|---:|---:|"]
        L += [f"| {p['subject_id']} | {p['n_windows']} | {p['accuracy']:.4f} | {p['macro_f1']:.4f} |"
              for p in r["per_subject"]]
        L += ["", f"mean={s['mean_subject_macro_f1']:.4f} std={s['std_subject_macro_f1']:.4f} "
                  f"min={s['min_subject_macro_f1']:.4f} (subject {s['worst_subject_id']}) "
                  f"max={s['max_subject_macro_f1']:.4f} (subject {s['best_subject_id']})", ""]
    L += ["## Pareto dominance (validation macro-F1 vs cost)", ""]
    for axis, res in out["pareto"].items():
        L.append(f"- vs {axis}: " + "; ".join(
            f"{n}: {'dominated by ' + r['dominated_by'] if r['dominated'] else 'non-dominated'}" for n, r in res.items()))
    path.write_text("\n".join(L) + "\n")


def plot_all(complexity, mean_f1, per_subject, fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for axis, xl, fn in (("macs", "MAC / window", "pareto_macs.png"),
                         ("params", "Trainable parameters", "pareto_params.png")):
        fig, ax = plt.subplots(figsize=(6.5, 5))
        for n in mean_f1:
            ax.scatter(complexity[n][axis], mean_f1[n], s=60)
            ax.annotate(n, (complexity[n][axis], mean_f1[n]), textcoords="offset points", xytext=(6, 4))
        ax.set_xlabel(xl)
        ax.set_ylabel("Mean validation macro-F1")
        ax.set_title(f"Validation macro-F1 vs {xl}")
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(fig_dir / fn, dpi=140)
        plt.close(fig)
    if per_subject:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        w = 0.8 / len(per_subject)
        for i, (n, r) in enumerate(per_subject.items()):
            rows = r["per_subject"]
            ax.bar([j + i * w for j in range(len(rows))], [p["macro_f1"] for p in rows], w, label=n)
        ax.set_xticks([j + 0.4 - w / 2 for j in range(len(rows))])
        ax.set_xticklabels([f"subj {p['subject_id']}" for p in rows])
        ax.set_ylabel("Macro-F1")
        ax.set_title("Per-validation-subject macro-F1")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "per_subject_macro_f1.png", dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
