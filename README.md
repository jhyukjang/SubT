# SubT: Subspace Tuning for Few-shot Generalization of Audio-Language Models

> **[EMNLP 2026 Findings] Official implementation of "Subspace Tuning for Few-shot Generalization of Audio-Language Models"**

SubT adapts audio-language models to downstream audio classification tasks by tuning a compact subspace of text features. This repository provides the code and scripts for few-shot and base-to-novel evaluation with PENGI.

## Installation

Create a Python 3.8 environment and install the required packages:

```bash
conda create -n subt python=3.8
conda activate subt
pip install -r requirements.txt
```

## Pretrained Model

Download the pretrained [PENGI `base.pth` checkpoint](https://zenodo.org/records/8387083/files/base.pth) and place it at:

```text
pengi/configs/base.pth
```

The expected directory structure is:

```text
subt/
├── main.py
├── pengi/
│   └── configs/
│       ├── base.yml
│       └── base.pth
├── scripts/
├── trainers/
└── utils/
```

## Datasets

We use the same 11 audio classification datasets and data splits as [PALM](https://github.com/asif-hanif/palm). Please follow the [PALM dataset preparation instructions](https://github.com/asif-hanif/palm/blob/main/DATASETS.md) to download and preprocess them.

| Dataset                                                                                                     | Task                              | Classes |
| :---------------------------------------------------------------------------------------------------------- | :-------------------------------- | ------: |
| [Beijing-Opera](https://compmusic.upf.edu/bo-perc-dataset)                                                   | Instrument classification         |       4 |
| [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)                                                 | Emotion recognition               |       6 |
| [ESC-50](https://github.com/karolpiczak/ESC-50)                                                              | Sound event classification        |      50 |
| [ESC-50 Actions](https://github.com/karolpiczak/ESC-50)                                                      | Sound event classification        |      10 |
| [GTZAN Music Genre](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) | Music genre classification        |      10 |
| [NSynth Instruments](https://magenta.tensorflow.org/datasets/nsynth)                                         | Instrument classification         |      10 |
| [RAVDESS](https://zenodo.org/records/1188976)                                                                | Emotion recognition               |       8 |
| [SESA](https://zenodo.org/records/3519845)                                                                   | Surveillance sound classification |       4 |
| [TUT Acoustic Scenes 2017](https://zenodo.org/records/400515)                                                | Acoustic scene classification     |      15 |
| [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html)                                       | Sound event classification        |      10 |
| [VocalSound](https://github.com/YuanGongND/vocalsound)                                                       | Vocal sound classification        |       6 |

Place the prepared datasets in a directory named `Audio-Datasets` at the repository root:

```text
subt/
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


## Acknowledgements

This repository was developed with reference to the official [PALM](https://github.com/asif-hanif/palm), including its PENGI-based experimental framework and dataset preparation pipeline. Many thanks to the authors for generously sharing their codes!
