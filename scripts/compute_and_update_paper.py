#!/usr/bin/env python3
"""
Read MLP probe results and LoRA-Large results, then patch the paper LaTeX.
Called by run_overnight.sh after all compute finishes.
"""
import json
import re
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROBE_DIR = REPO / "results" / "v21_etth1_mlp"
LORA_DIRS = [
    REPO / "results" / "v13_lora_large_lr1e-5" / "seed123",
    REPO / "results" / "v13_lora_large_lr1e-5" / "seed456",
    REPO / "results" / "v12_lora_large_hp" / "lr1e-5",
    REPO / "results" / "v21_lora_large_k5" / "seed303",
    REPO / "results" / "v21_lora_large_k5" / "seed789",
]
ANALYSIS_TEX = REPO / "paper_8" / "sections" / "07_analysis.tex"
CONCLUSION_TEX = REPO / "paper_8" / "sections" / "08_conclusion.tex"
INTRO_TEX = REPO / "paper_8" / "sections" / "01_introduction.tex"
RESPONSE_LETTER = REPO / "paper_8" / "response_letter.md"


# ─────────────────────────────────────────────────────────────────────
# 1. MLP probe results
# ─────────────────────────────────────────────────────────────────────

def load_mlp_results():
    seeds = [42, 101, 123, 202, 303, 456, 777, 789, 888, 999]
    zs_file = PROBE_DIR / "zeroshot_mlp.json"
    if not zs_file.exists():
        print(f"ERROR: ZS MLP file not found: {zs_file}")
        return None

    zs = json.loads(zs_file.read_text())
    # Structure: {head_type: {probe_type: {k_depth: r2}}}
    # or flat depending on reprobe script output format
    print("ZS keys:", list(zs.keys())[:5])

    results = {}
    for seed in seeds:
        f = PROBE_DIR / f"seed{seed}_mlp.json"
        if not f.exists():
            print(f"  Missing: seed{seed}")
            continue
        results[seed] = json.loads(f.read_text())
    return zs, results


def compute_mlp_delta_r2(zs, ft_results, k_depth=5):
    """Compute ΔR² = R²(FT) - R²(ZS) for the forecast96 head at given MLP depth."""
    # Try to navigate the JSON structure - need to find the right key path
    def get_r2(d, k):
        # Try multiple possible structures from reprobe script
        for path in [
            ["forecast96", "mlp", k],
            ["forecast96", f"mlp_k{k}"],
            [f"forecast96_mlp_k{k}"],
            ["mlp", "forecast96", k],
        ]:
            try:
                v = d
                for p in path:
                    v = v[p]
                return float(v)
            except (KeyError, TypeError):
                pass
        # Flat key search
        for key, val in d.items():
            if "forecast96" in str(key) and "mlp" in str(key) and str(k) in str(key):
                try:
                    return float(val)
                except (TypeError, ValueError):
                    pass
        return None

    zs_r2 = get_r2(zs, k_depth)
    if zs_r2 is None:
        print(f"  Could not extract ZS R²(k={k_depth}) from {list(zs.keys())}")
        return None, None, None

    print(f"ZS R²(MLP k={k_depth}, forecast96) = {zs_r2:.4f}")

    deltas = {}
    for seed, d in ft_results.items():
        ft_r2 = get_r2(d, k_depth)
        if ft_r2 is not None:
            deltas[seed] = ft_r2 - zs_r2
            print(f"  seed{seed}: R²(FT)={ft_r2:.4f}  ΔR²={deltas[seed]:+.4f}")
        else:
            print(f"  seed{seed}: could not extract R²(FT)")

    if not deltas:
        return None, None, None

    vals = list(deltas.values())
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    n_pos = sum(1 for v in vals if v > 0)
    print(f"\nMLP k={k_depth}: ΔR² = {mean:+.3f}±{std:.3f} ({n_pos}/{len(vals)} positive)")
    return mean, std, n_pos, len(vals), deltas


# ─────────────────────────────────────────────────────────────────────
# 2. LoRA-Large k=5 results
# ─────────────────────────────────────────────────────────────────────

def load_lora_results():
    forgetting_vals = []
    cka_vals = []
    for d in LORA_DIRS:
        for f in d.glob("*.json"):
            data = json.loads(f.read_text())
            if data.get("model_size") == "large" and data.get("condition") == "E":
                lr = data.get("lr", data.get("learning_rate", None))
                # All these dirs are LR=1e-5 runs
                forg = data.get("forgetting_pct")
                cka = data.get("final_cka")
                if forg is not None:
                    forgetting_vals.append(forg)
                    print(f"  {f.name}: seed={data.get('seed')} forg={forg:.3f}% CKA={cka:.4f}")
                if cka is not None:
                    cka_vals.append(cka)

    if not forgetting_vals:
        return None
    mean_f = statistics.mean(forgetting_vals)
    std_f = statistics.stdev(forgetting_vals) if len(forgetting_vals) > 1 else 0.0
    print(f"\nLoRA-Large LR=1e-5 k={len(forgetting_vals)}: forg={mean_f:+.1f}±{std_f:.1f}%")
    return mean_f, std_f, len(forgetting_vals)


# ─────────────────────────────────────────────────────────────────────
# 3. Patch paper LaTeX
# ─────────────────────────────────────────────────────────────────────

def patch_analysis_tex(mlp_mean, mlp_std, mlp_n_pos, mlp_n_total, mlp_k=5):
    """Replace the PLACEHOLDER in §7 with actual MLP results."""
    text = ANALYSIS_TEX.read_text()

    if "PLACEHOLDER: insert per-seed MLP" not in text:
        print("analysis.tex: placeholder already replaced, skipping")
        return

    n_pos = mlp_n_pos
    n_total = mlp_n_total
    sign_word = "positive" if n_pos > n_total / 2 else "negative"
    hypothesis_supported = n_pos >= 6  # majority positive = hypothesis supported

    if hypothesis_supported:
        interpretation = (
            f"supporting the non-linear restructuring hypothesis: "
            f"the fine-tuned ETTh1 encoder encodes features that are "
            f"linearly inseparable but non-linearly accessible by an MLP."
        )
    else:
        interpretation = (
            f"failing to rescue the Ridge signal. "
            f"The non-linear restructuring hypothesis is not supported; "
            f"an alternative interpretation is that the fine-tuned encoder "
            f"encodes objective-specific features (96-step NLL curvature) "
            f"that improve MSE but are not decodable by any probe we tested."
        )

    replacement = (
        f"MLP probes ($k={mlp_k}$ hidden layers, 64 units each) on the 10 ETTh1 CUDA "
        f"encoders give $\\Delta R^2{{=}}{mlp_mean:+.2f}{{\\pm}}{mlp_std:.2f}$ "
        f"({n_pos}/{n_total} positive), "
        + interpretation
    )

    old_block = (
        "To test the non-linear restructuring hypothesis directly, we run an\n"
        "MLP probe ($k{=}1,2,5$ hidden layers, 64 units each,\n"
        "\\texttt{reprobe\\_saved\\_encoders.py}) on the 10 ETTh1 CUDA encoders\n"
        "and compare to the zero-shot MLP baseline.\n"
        "% PLACEHOLDER: insert per-seed MLP ΔR² table and interpretation after results arrive"
    )
    new_block = (
        "To test the non-linear restructuring hypothesis directly, we run an\n"
        "MLP probe ($k{=}1,2,5$ hidden layers, 64 units each,\n"
        "\\texttt{reprobe\\_saved\\_encoders.py}) on the 10 ETTh1 CUDA encoders\n"
        "and compare to the zero-shot MLP baseline.\n"
        + replacement
    )

    if old_block in text:
        text = text.replace(old_block, new_block)
        ANALYSIS_TEX.write_text(text)
        print(f"analysis.tex: updated MLP probe paragraph (k={mlp_k}, {n_pos}/{n_total})")
    else:
        print("WARNING: could not find placeholder block in analysis.tex — manual update needed")
        print("Replacement text:")
        print(replacement)


def patch_conclusion_tex(mean_f, std_f, k):
    """Update LoRA-Large numbers in §8 practitioner summary."""
    text = CONCLUSION_TEX.read_text()
    # Current text uses k=3 stats: −8.5±0.8%
    old = f"forg.$=-$8.5$\\pm$0.8\\%, 3 seeds)"
    new = f"forg.$={mean_f:+.1f}{{\\pm}}{std_f:.1f}$\\%, {k} seeds)"
    if old in text:
        text = text.replace(old, new)
        CONCLUSION_TEX.write_text(text)
        print(f"conclusion.tex: updated LoRA-Large forg to {mean_f:+.1f}±{std_f:.1f}% (k={k})")
    else:
        # Try alternate format
        old2 = "forg.$=-$8.5$\\pm$0.8\\%"
        if old2 in text:
            new2 = f"forg.$={mean_f:+.1f}{{\\pm}}{std_f:.1f}$\\%"
            text = text.replace(old2, new2)
            CONCLUSION_TEX.write_text(text)
            print(f"conclusion.tex: updated (alt match)")
        else:
            print(f"WARNING: could not find LoRA-Large forg string in conclusion.tex")
            print(f"  New values: {mean_f:+.1f}±{std_f:.1f}% (k={k})")


def patch_intro_tex(mean_f, std_f, k):
    """Update LoRA-Large numbers in §1 contribution 2."""
    text = INTRO_TEX.read_text()
    old = "forg.$=-$8.5$\\pm$0.8\\%,\n    3 seeds)"
    new = f"forg.$={mean_f:+.1f}{{\\pm}}{std_f:.1f}$\\%,\n    {k} seeds)"
    if old in text:
        text = text.replace(old, new)
        INTRO_TEX.write_text(text)
        print(f"intro.tex: updated LoRA-Large forg")
    else:
        # Try without newline
        old2 = "forg.$=-$8.5$\\pm$0.8\\%, 3 seeds)"
        new2 = f"forg.$={mean_f:+.1f}{{\\pm}}{std_f:.1f}$\\%, {k} seeds)"
        if old2 in text:
            text = text.replace(old2, new2)
            INTRO_TEX.write_text(text)
            print(f"intro.tex: updated LoRA-Large forg (alt match)")
        else:
            print(f"WARNING: could not find LoRA-Large forg string in intro.tex")


def patch_response_letter(mlp_mean, mlp_std, mlp_n_pos, mlp_n_total,
                          lora_mean, lora_std, lora_k):
    """Fill in PLACEHOLDERs in response letter."""
    text = RESPONSE_LETTER.read_text()
    n = mlp_n_total

    mlp_result_line = (
        f"ΔR²(MLP k=5) = {mlp_mean:+.2f}±{mlp_std:.2f} ({mlp_n_pos}/{n} positive)"
    )
    lora_result_line = (
        f"LoRA-Large LR=1e-5: forgetting = {lora_mean:+.1f}±{lora_std:.1f}% (k={lora_k} seeds)"
    )

    # Replace MLP PLACEHOLDER
    for placeholder in [
        "PLACEHOLDER: insert MLP ΔR² result here",
        "PLACEHOLDER: MLP probe result",
        "[MLP probe result PLACEHOLDER]",
        "MLP probe results: PLACEHOLDER",
    ]:
        if placeholder in text:
            text = text.replace(placeholder, mlp_result_line)
            break

    # Replace LoRA PLACEHOLDER
    for placeholder in [
        "PLACEHOLDER: insert k=5 LoRA-Large result here",
        "PLACEHOLDER: LoRA-Large k=5 result",
        "[LoRA-Large k=5 PLACEHOLDER]",
        "LoRA-Large result: PLACEHOLDER",
    ]:
        if placeholder in text:
            text = text.replace(placeholder, lora_result_line)
            break

    RESPONSE_LETTER.write_text(text)
    print("response_letter.md: updated with compute results")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Loading MLP probe results...")
    print("=" * 60)
    mlp_data = load_mlp_results()

    print("\n" + "=" * 60)
    print("Loading LoRA-Large results...")
    print("=" * 60)
    lora_data = load_lora_results()

    mlp_ok = mlp_data is not None and len(mlp_data[1]) >= 8  # at least 8/10 seeds
    lora_ok = lora_data is not None and lora_data[2] >= 4  # at least k=4

    if mlp_ok:
        zs, ft_results = mlp_data
        result = compute_mlp_delta_r2(zs, ft_results, k_depth=5)
        if result[0] is not None:
            mlp_mean, mlp_std, mlp_n_pos, mlp_n_total, _ = result
            print(f"\nPatching §7 with MLP results...")
            patch_analysis_tex(mlp_mean, mlp_std, mlp_n_pos, mlp_n_total, mlp_k=5)
        else:
            print("Could not parse MLP results — check JSON structure manually")
    else:
        print(f"MLP results incomplete: {len(mlp_data[1]) if mlp_data else 0}/10 seeds")

    if lora_ok:
        lora_mean, lora_std, lora_k = lora_data
        print(f"\nPatching §8 and §1 with LoRA-Large k={lora_k} results...")
        patch_conclusion_tex(lora_mean, lora_std, lora_k)
        patch_intro_tex(lora_mean, lora_std, lora_k)
    else:
        print(f"LoRA-Large results incomplete: {lora_data[2] if lora_data else 0} seeds")

    if mlp_ok and lora_ok:
        patch_response_letter(
            mlp_mean, mlp_std, mlp_n_pos, mlp_n_total,
            lora_mean, lora_std, lora_k,
        )
        print("\nAll patches applied. Run pdflatex to verify.")
    else:
        print("\nSome results missing — partial update only.")
