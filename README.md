# SubT: Subspace Tuning for Few-shot Generalization of Audio-Language Models

<p align="center">
  <a href="https://arxiv.org/abs/2606.18560"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg" alt="arXiv Paper"></a>
  <br><br>
  <a href="https://jhyukjang.github.io/">Jaehyuk Jang</a>, Kangwook Ko, <a href="https://wonjun-lee1009.github.io/">Wonjun Lee</a>, <a href="https://cilabs.kaist.ac.kr/members">Changick Kim</a>
  <br>
  KAIST
</p>

## Installation

Create a Python 3.8 environment and install the required packages:

```bash
conda create -n subt python=3.8
conda activate subt
pip install -r requirements.txt
```

## Pretrained Model

Download the pretrained [PENGI checkpoint](https://zenodo.org/records/8387083/files/base.pth) `base.pth` and place it at:

```text
pengi/configs/base.pth
```

The expected directory structure is:

```text
SubT/
├── pengi/
│   └── configs/
│       └── base.pth
```

## Datasets

We use the same 11 audio classification datasets and data splits as [PALM](https://github.com/asif-hanif/palm). Please follow the [PALM dataset preparation instructions](https://github.com/asif-hanif/palm/blob/main/DATASETS.md) to download and preprocess them.

Place the prepared datasets in a directory named `Audio-Datasets` at the repository root:

```text
SubT/
└── Audio-Datasets/
    ├── Beijing-Opera/
    ├── CREMA-D/
    ├── ESC50/
    ├── ESC50-Actions/
    ├── GT-Music-Genre/
    ├── NS-Instruments/
    ├── RAVDESS/
    ├── SESA/
    ├── TUT2017/
    ├── UrbanSound8K/
    └── VocalSound/
```

## Running Experiments

Run commands from the repository root.

### Base-to-Novel Generalization

```bash
bash scripts/run_all_datasets_subt.sh
```

## Citation

If you find this repository useful, please cite our paper.

```bibtex
@article{jang2026constraining,
  title={SubT: Subspace Tuning for Few-shot Generalization of Audio-Language Models},
  author={Jang, Jaehyuk and Ko, Kangwook and Lee, Wonjun and Kim, Changick},
  journal={arXiv preprint arXiv:2606.18560},
  year={2026},
}
```

## Acknowledgements

This repository was developed with reference to the official [PALM](https://github.com/asif-hanif/palm), including its PENGI-based experimental framework and dataset preparation pipeline. Many thanks to the authors for generously sharing their codes!
