# CSA-6G

<!-- Add a short description of the project here, in your own words. -->

Code and data for the paper "CSA-6G: Confidentiality-Preserving Slice Admission and Defense-in-Depth Allocation Toward 6G" (Aslı Kul, Sevtap Duman, Hasan Bulut; Ege University).

## Contents

| Folder | Contents |
|---|---|
| `src/csa6g/` | Prototype |
| `tests/` | Test suite (140 tests) |
| `experiments/` | Experiments 1–8 and 4b; `run_all.py` runs them all |
| `data/reported_run/` | Results reported in the paper, with environment record and logs |

## Setup

Python 3.12 was used for the reported run (`data/reported_run/environment/pip_freeze.txt`).

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```sh
make test       # run the tests
make quick      # reduced-scale run of every experiment
make all        # full-scale run
```

`CSA6G_OUT=<folder> make all` writes a run to its own folder.

Experiment 5 in `data/reported_run` was rerun separately on the same host (`environment/exp5_*`).

## License

<!-- Add the license name here once LICENSE is added. -->

## Citation

<!-- Add the paper citation and Zenodo DOI here. -->
