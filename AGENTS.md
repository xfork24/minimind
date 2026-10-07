# Repository Guidelines

## Project Structure & Module Organization

- `model/` contains the MiniMind transformer, LoRA implementation, and tokenizer assets.
- `trainer/` contains pretraining, SFT, distillation, preference optimization, and agent training scripts; shared helpers live in `trainer_utils.py`.
- `dataset/lm_dataset.py` implements dataset loaders. Download JSONL training data into `dataset/`; see `dataset/dataset.md` for details.
- `eval_llm.py` provides CLI inference. `scripts/` contains model conversion, tool-call evaluation, API serving, and the Streamlit demo.
- `images/` stores documentation assets. `README.md` and `README_en.md` describe workflows in Chinese and English.
- Generated weights go into `out/`; resumable training state goes into `checkpoints/`.

## Build, Test, and Development Commands

Run from the repository root:

- `python -m pip install -r requirements.txt`: install dependencies. Install a PyTorch build appropriate for your hardware separately; its requirement is commented out.
- `python -m compileall model trainer dataset scripts eval_llm.py`: check Python syntax.
- `(cd trainer && python train_pretrain.py)`: pretrain using `dataset/pretrain_t2t_mini.jsonl`.
- `(cd trainer && python train_full_sft.py)`: fine-tune pretrained weights on instruction data.
- `python eval_llm.py --load_from ./model --weight full_sft`: evaluate native weights from `out/`.
- `(cd scripts && streamlit run web_demo.py)`: launch the web demo after placing a Transformers-format model directory under `scripts/`.

Training and inference require the corresponding datasets or weights. Use each script's `--help` for configuration; preserve working-directory assumptions in relative paths.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` for functions, variables, and modules, and `PascalCase` for classes. Follow neighboring code and keep changes focused. Name training entry points `train_<stage>.py`; preserve checkpoint names such as `full_sft_768.pth`. No formatter or linter configuration is currently checked in.

## Testing Guidelines

There is no automated test suite or coverage threshold. Run syntax checks and exercise the affected workflow with a small dataset or compatible checkpoint. For training changes, verify finite loss, checkpoint saving, and resume behavior. For inference changes, check generation; use `(cd scripts && python eval_toolcall.py --weight full_sft)` for tool-call changes. Include reproducible commands and results in the PR.

## Commit & Pull Request Guidelines

History uses prefixes such as `fix:`, `[fix]`, and `[update]`. Keep PRs focused; explain the problem, resulting behavior, related issues, and validation, including hardware and model configuration for training changes. Include screenshots for UI changes and update relevant README instructions. Follow `CODE_OF_CONDUCT.md`.

## Configuration & Artifacts

Keep API credentials, downloaded datasets, and generated checkpoints out of commits. `out/` is ignored; verify other artifact directories before staging.
