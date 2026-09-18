# Classified-Study
Supplementary code for the paper:
**Classification of Natural Earthquakes and Artificial Explosion Events Based on a Deep Learning Hybrid Architecture Combining 1D-CNN and Transformer Models**

This repository contains the full Python implementation of five deep learning models proposed and compared in this paper, including data preprocessing, model construction, training, validation and evaluation scripts for seismic waveform discrimination.

## Requirements
The code is implemented with Python. Install all dependencies via:
```
pip install -r requirements.txt
```

Main packages:

- python >=3.9
- numpy
- obspy
- tensorflow
- scikit-learn
- scipy
- matplotlib

## Project Structure

```
├── data_preprocess.py   # Seismic waveform reading and preprocessing
├── model_1dcnn.py       # 1D-CNN model
├── model_transformer.py # Transformer model
├── model_hybrid.py      # Proposed 1D-CNN + Transformer hybrid model
├── model_mlp.py
├── model_svm.py
├── train.py             # Training script
├── evaluate.py          # Model evaluation and metrics calculation
├── utils.py
├── requirements.txt
└── demo/                # Small demo waveform samples
```

## How to run

1. Prepare waveform data (or use demo samples in `/demo`)
2. Modify the data path in `train.py`
3. Run training:

```
python train.py
```

4. Evaluate model performance:

```
python evaluate.py
```

## Data statement

Small demo waveform samples are provided to verify the code pipeline.
The full raw seismic waveform dataset cannot be publicly released due to project restrictions. Researchers can contact the corresponding author for access to the full dataset upon reasonable request.

## Citation

If you use this code in your work, please cite our paper once published.
