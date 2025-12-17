# Experiment Configuration Guide

## Overview
This directory contains YAML configuration files for different experiments. Each configuration file defines the parameters for training and evaluation.

## Usage

### Loading Configuration Files

```python
import yaml

with open("experiments/gan_augmented.yaml") as f:
    cfg = yaml.safe_load(f)
```

### Configuration Structure

The loaded configuration is a dictionary with the following structure:

```python
cfg = {
    "experiment_name": "gan_augmented",
    "dataset": {
        "type": "augmented",
        "synthetic_ratio": 1.0
    },
    "model": {
        "name": "resnet50",
        "pretrained": True
    },
    "training": {
        "epochs": 50,
        "batch_size": 32,
        "lr": 0.0001
    }
}

