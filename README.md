# Brainbox 15

## System Information

| Tool | Version |
| -- | -- |
| Windows [10] | 10.0.19041 Build 19041 |
| Python | 3.8.10 [MSC v.1928 64 bit (AMD64)] |

## Directory Tour

root \
├── accuracy_testing.py _(see below)_ \
├── Fourier.py _(physics functions)_ \
├── SimpleClassifier.py _(extra functions)_ \
├── training_data \
│   ├── audio_waves \
│   ├── spiker_filtered _(filtered waves)_ \
│   └── spiker_waves _(unfiltered waves)_ \
└── spaceinvaders.py _(main game program)_ \
└── model.sav _(trained model to be loaded)_ \
└── scaler.sav _(associated model scaler)_ \
└── assets _(spaceinvaders images)_ \
└── requirements.txt _(please read for more details)_

## Instructions

First, install dependencies: `python3 -m pip install -r requirements.txt`.

Run `python3 spaceinvaders.py` to start the game.

> Note: You will need a spikerbox plugged in to start the game. 2018 model tends to work better (from our experiences).

