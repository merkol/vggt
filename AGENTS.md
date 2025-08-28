# Repository Guidelines

## Project Structure & Module Organization
- `vggt/`: Core library (models, layers, heads, utils). Import from `vggt.*`.
- `training/`: Fine‑tuning code, configs, and launch scripts. See `training/README.md`.
- `examples/`: Sample images/videos for quick runs and visualization.
- `docs/`: Extra documentation (e.g., package install options).
- Root scripts: `demo_gradio.py`, `demo_viser.py`, `demo_colmap.py` for demos and export.

## Build, Test, and Development Commands
- Install runtime deps: `pip install -r requirements.txt`.
- Editable install: `pip install -e .` (develop against `vggt/`).
- Demos: `python demo_gradio.py` (web UI), `python demo_viser.py --image_folder path/to/images`.
- COLMAP export: `python demo_colmap.py --scene_dir /YOUR/SCENE_DIR [--use_ba]`.
- Training (DDP): `cd training && torchrun --nproc_per_node=4 launch.py`.
- Optional demo deps: `pip install -r requirements_demo.txt`.

## Coding Style & Naming Conventions
- Language: Python ≥ 3.10; follow PEP 8, 4‑space indentation.
- Names: `snake_case` for modules/functions, `PascalCase` for classes, `CONSTANT_CASE` for constants.
- Type hints encouraged in public APIs; keep docstrings concise with usage notes.
- Keep functions focused; prefer small, testable units in `vggt/*`.

## Testing Guidelines
- Current repo has no formal test suite. Add `pytest` tests under `tests/` mirroring package paths (e.g., `tests/vggt/test_geometry.py`).
- Name tests `test_*.py`; write fast, deterministic tests (CPU‑only when possible).
- For smoke tests, run demos on tiny inputs from `examples/`.

## Commit & Pull Request Guidelines
- Commits: short, imperative subject (≤ 72 chars), present tense. Examples: "fix dataloader seed", "update README", "relicense for commercial use".
- Scope commits logically (one change per commit) and include brief rationale in body if non‑obvious.
- PRs: include problem statement, approach, minimal repro or screenshots (for demos), and links to related issues.
- Checklist: passes basic runs (`demo_*`, training launch if affected), docs updated, no large binaries or datasets committed.

## Security & Configuration Tips
- Do not commit checkpoints, datasets, or private keys; keep paths in YAML under `training/config/`.
- Large downloads (weights) are fetched via Hugging Face; prefer environment variables or config files for local paths.
