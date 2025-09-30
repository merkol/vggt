import re
import csv
import torch
from vggt.models.vggt import VGGT

# ------------------- toggles -------------------
SHOW_PER_BLOCK = False  # set True to print each frame/global block params
EXPORT_CSV = False  # set True to write a CSV to ./vggt_param_breakdown.csv
CSV_PATH = "vggt_param_breakdown.csv"
# ------------------------------------------------


def nparams(module, recurse=True):
    return sum(p.numel() for p in module.parameters(recurse=recurse))


def human(n: int) -> str:
    if n >= 10**9:
        return f"{n / 1e9:.3f}B"
    if n >= 10**6:
        return f"{n / 1e6:.3f}M"
    if n >= 10**3:
        return f"{n / 1e3:.3f}K"
    return str(n)


# Scope-aware category matcher (by qualified module path)
def in_path(name: str, key: str) -> bool:
    return (name == key) or (name.startswith(key + "."))


def scope_sum_own(model, scope_prefix_pred):
    """
    Sum OWN parameters of modules whose qualified name matches a predicate.
    Because we sum 'own' parameters, adding across multiple modules does not double count.
    """
    total = 0
    for name, mod in model.named_modules():
        if scope_prefix_pred(name, mod):
            own = sum(p.numel() for p in mod.parameters(recurse=False))
            total += own
    return total


# ---------- load official checkpoint ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()

total = nparams(model)
print(f"\nTOTAL PARAMS: {total:,}  ({human(total)})")

# ---------- top-level split ----------
child_counts = []
for name, child in model.named_children():
    c = nparams(child)
    child_counts.append((name, c, 100.0 * c / total))
child_counts.sort(key=lambda x: x[1], reverse=True)

print("\nTop-level modules:")
for name, c, pct in child_counts:
    print(f"{name:22s} {human(c):>12}  {pct:6.2f}%")

other = total - sum(c for _, c, _ in child_counts)
if other > 0:
    print(f"{'other(top-level)':22s} {human(other):>12}  {100.0 * other / total:6.2f}%")

# ---------- aggregator breakdown (scope-aware) ----------
if not hasattr(model, "aggregator"):
    raise AttributeError(
        "Model has no 'aggregator' attribute; VGGT API may have changed."
    )

agg = model.aggregator
agg_total_recursive = nparams(agg)  # full aggregator params (recursive)
agg_root_own = sum(
    p.numel() for p in agg.parameters(recurse=False)
)  # camera/register tokens live here

# Patch embed (tokenizer / ViT stem) lives under aggregator.patch_embed
patch_embed_total = (
    nparams(getattr(agg, "patch_embed")) if hasattr(agg, "patch_embed") else 0
)

# Attention (all params whose path contains ".attn" within aggregator)
attn_total_own = scope_sum_own(
    agg, lambda name, mod: in_path(name, "attn") or (".attn" in name)
)

# MLP/FFN (paths containing ".mlp" or ".ffn")
mlp_total_own = scope_sum_own(
    agg, lambda name, mod: (".mlp" in name) or (".ffn" in name)
)

# Norms (paths containing ".norm")
norm_total_own = scope_sum_own(agg, lambda name, mod: ".norm" in name)

# Positional / RoPE
rope_total_own = scope_sum_own(
    agg, lambda name, mod: ("rope" in name.lower()) or (".pos" in name.lower())
)

# Everything unaccounted for inside aggregator after categories above
accounted_agg = (
    patch_embed_total
    + attn_total_own
    + mlp_total_own
    + norm_total_own
    + rope_total_own
    + agg_root_own
)
agg_other = agg_total_recursive - accounted_agg
if agg_other < 0:
    # Guard small negative from heuristics / name mismatches
    agg_other = 0

print("\nAggregator breakdown:")
print(
    f"{'patch_embed (encoder)':22s} {human(patch_embed_total):>12}  {100.0 * patch_embed_total / agg_total_recursive:6.2f}% of agg   {100.0 * patch_embed_total / total:6.2f}% of total"
)
print(
    f"{'attention (own)':22s} {human(attn_total_own):>12}  {100.0 * attn_total_own / agg_total_recursive:6.2f}% of agg   {100.0 * attn_total_own / total:6.2f}% of total"
)
print(
    f"{'mlp/ffn (own)':22s} {human(mlp_total_own):>12}  {100.0 * mlp_total_own / agg_total_recursive:6.2f}% of agg   {100.0 * mlp_total_own / total:6.2f}% of total"
)
print(
    f"{'norms (own)':22s} {human(norm_total_own):>12}  {100.0 * norm_total_own / agg_total_recursive:6.2f}% of agg   {100.0 * norm_total_own / total:6.2f}% of total"
)
print(
    f"{'rope/pos (own)':22s} {human(rope_total_own):>12}  {100.0 * rope_total_own / agg_total_recursive:6.2f}% of agg   {100.0 * rope_total_own / total:6.2f}% of total"
)
print(
    f"{'special tokens (own)':22s} {human(agg_root_own):>12}  {100.0 * agg_root_own / agg_total_recursive:6.2f}% of agg   {100.0 * agg_root_own / total:6.2f}% of total"
)
print(
    f"{'other (agg)':22s} {human(agg_other):>12}  {100.0 * agg_other / agg_total_recursive:6.2f}% of agg   {100.0 * agg_other / total:6.2f}% of total"
)

# ---------- heads (top-level) ----------
heads_total = 0
heads_detail = {}
for name, c, _ in child_counts:
    if "head" in name.lower():
        heads_detail[name] = c
        heads_total += c

if heads_detail:
    print("\nHeads:")
    for k, v in sorted(heads_detail.items(), key=lambda x: x[1], reverse=True):
        print(f"{k:22s} {human(v):>12}  {100.0 * v / total:6.2f}% of total")
else:
    print("\nHeads: none found at top level (API may have changed).")

# ---------- E/D/A-style disjoint split ----------
# Attention = all attention own-params in the aggregator (attn_total_own).
# Heads     = sum of all top-level *head* modules.
# Encoder*  = everything else (tokenizer + non-attn parts of aggregator + any residual top-level).
attention_only = attn_total_own
decoder_heads = heads_total
encoder_star = total - attention_only - decoder_heads

print("\nE/D/A-style disjoint split (sums to TOTAL):")
print(
    f"{'Encoder*':22s} {human(encoder_star):>12}  {100.0 * encoder_star / total:6.2f}%"
)
print(
    f"{'Decoder (heads)':22s} {human(decoder_heads):>12}  {100.0 * decoder_heads / total:6.2f}%"
)
print(
    f"{'Attention (in agg)':22s} {human(attention_only):>12}  {100.0 * attention_only / total:6.2f}%"
)
print(
    "(*) Encoder = everything not in heads or attention; includes patch_embed and non-attention parts of the aggregator."
)

# ---------- optional: per-block breakdown ----------
if SHOW_PER_BLOCK and hasattr(agg, "depth"):
    print("\nPer-block params:")
    depth = getattr(agg, "depth")
    # safety: frame_blocks/global_blocks may be ModuleList
    fb, gb = getattr(agg, "frame_blocks", None), getattr(agg, "global_blocks", None)
    if fb is not None and gb is not None:
        for i in range(depth):
            fb_i = nparams(fb[i]) if i < len(fb) else 0
            gb_i = nparams(gb[i]) if i < len(gb) else 0
            print(
                f"block {i:02d}  frame={human(fb_i):>8}  global={human(gb_i):>8}  total={human(fb_i + gb_i):>8}"
            )
    else:
        print("No frame_blocks/global_blocks found (API may have changed).")

# ---------- optional: CSV export ----------
if EXPORT_CSV:
    rows = []
    rows.append(["section", "name", "params"])
    rows.append(["total", "TOTAL", total])
    # top-level
    for name, c, _ in child_counts:
        rows.append(["top_level", name, c])
    if other > 0:
        rows.append(["top_level", "other(top-level)", other])

    # aggregator categories
    rows.append(["aggregator", "agg_total_recursive", agg_total_recursive])
    rows.append(["aggregator", "patch_embed", patch_embed_total])
    rows.append(["aggregator", "attention_own", attn_total_own])
    rows.append(["aggregator", "mlp_own", mlp_total_own])
    rows.append(["aggregator", "norm_own", norm_total_own])
    rows.append(["aggregator", "rope_pos_own", rope_total_own])
    rows.append(["aggregator", "special_tokens_own", agg_root_own])
    rows.append(["aggregator", "other_agg", agg_other])

    # heads detail
    for k, v in heads_detail.items():
        rows.append(["heads", k, v])

    # E/D/A
    rows.append(["EDA", "Encoder*", encoder_star])
    rows.append(["EDA", "Decoder(heads)", decoder_heads])
    rows.append(["EDA", "Attention(in agg)", attention_only])

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"\nCSV written → {CSV_PATH}")
