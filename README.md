# Diploma Thesis - Computer Network Monitoring with Artificial Intelligence Methods

Author: Bc. Tomáš Rončák \
Supervisor: Assoc. Prof. Ing. Giang Nguyen Thu, PhD.

## Assignment

Počítačové siete sú neustále ohrozované novými druhmi kybernetických hrozieb. Jedným zo základných prvkov ochrany počítačovej siete je monitorovanie sietí. Bežným príkladom je systém detekcie prieniku. Takéto systémy trpia veľkým nedostatkom a to neschopnosťou detegovať nové druhy útokov, ktoré sa nenachádzajú v databázach známych útokov. Tento nedostatok vytvára potenciálne veľké ohrozenie pre počítačové siete, a preto je žiadané využívať také riešenia, ktoré dokážu detegovať aj nové druhy hrozieb. Jedným z nástrojov, ktoré sa snažia detegovať aj nové hrozby je detekcia anomálii pomocou strojového učenia. Témou diplomovej práce je monitorovanie sieťových protokolov pomocou vybraných architektúr hlbokého učenia a detekcia hrozieb v sieti. Vykonajte analýzu súčasného stavu problematiky monitorovania sietí s využitím metód umelej inteligencie. Navrhnite a implementujte softvérový prototyp s natrénovaným modelom hlbokého učenia, ktorý bude schopný rozpoznávať hrozby v počítačových sieťach z dát vybraných sieťových protokolov. Cieľom je vytvorenie takého riešenia, ktoré bude schopné vytvárať si komplexný obraz o stave siete a vyriešiť problémy spojené s reálnym nasadením do vysoko dynamickej sieťovej prevádzky. Zároveň je dôležité navrhnúť systém tak, aby dosahoval čo najlepšie výsledky pri zachovaní čo najnižšom počte falošných poplachov. Ako jeden z niekoľkých prostriedkov preskúmajte možnosť hybridného systému detekcie hrozieb. Podrobne vyhodnoťte prototyp pomocou dostupných metrík a porovnajte dosiahnuté výsledky s inými prístupmi. Definujte hranice použiteľnosti riešenia ako aj smer, ktorým by mal výskum v danej problematike pokračovať.

## Technologies

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://docs.python.org/3/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/api_docs/python/tf)
[![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io/api/)

## Prerequisites ✔️

- `Python 3.7+` installed on your machine: `sudo apt-get install python3-pip`
- `pip3` installed on your machine: `sudo pip3 install virtualenv`

## How to run 🏃 project

1. `virtualenv venv` - Create virtual python environment from root folder
2. `source venv/bin/activate` - Activate virtual environment
3. `python -m pip install --upgrade pip` Upgrade pip
4. `pip3 install -r requirements.txt` - Install packages
5. Based on your IDE, select this active virtual environment as interpreter

## How to stop 🛑 project

1. `deactivate` - Deactivate virtual environment

## Cheatsheet 📝

- `pip3 freeze > requirements.txt` - generate _requirements.txt_ file
- `pipreqs [<path>]` - generate shorter _requirements.txt_ file with only directly used libraries

## Dataset 📊
- [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

##  Project Organization
    ├── README.md
    ├── data
    │   ├── extracted_datasets                      <- Final datasets for anomaly modeling (selected attributes)
    │   │   ├── 180                                 <- Time series dataset extracted with 180 seconds sliding windows
    │   │   ├── 200
    │   │   ├── 220
    │   │   └── benign_180
    │   ├── original                                <- Original dataset
    │   │   ├── UNSW-NB15_1.csv
    │   │   ├── UNSW-NB15_2.csv
    │   │   ├── UNSW-NB15_3.csv
    │   │   └── UNSW-NB15_4.csv
    │   ├── preprocessed_category                   <- Final datasets for category modeling
    │   │   ├── dos_dataset.csv
    │   │   ├── exploits_dataset.csv
    │   │   ├── fuzzers_dataset.csv
    │   │   ├── generic_dataset.csv
    │   │   ├── reconnaissance_dataset.csv
    │   │   ├── test_dataset.csv
    │   │   ├── train_val_dataset.csv
    │   │   └── whole_dataset.csv
    │   └── processed_anomaly                       <- Dataset for anomaly modeling (all attributes)
    │       ├── 180
    │       ├── 200
    │       └── 220
    ├── models                                      <- Trained and serialized models
    │   └── models_1
    │       ├── anomaly_model
    │       ├── classification_model_bin
    │       └── classification_model_mult
    ├── notebooks
    │   ├── data_upsampling.ipynb
    │   └── eda.ipynb
    ├── requirements.txt
    └── src
        ├── config.py                               <- Configuration variables
        ├── constants.py
        ├── data
        │   ├── ClassificationDataHandler.py        <- Clasification data preprocessing
        │   ├── DataToTimeSeriesTransformator.py    <- Transformating classification data to time series
        │   ├── TimeSeriesDataFormatter.py          <- Formatting time series data for anomaly model
        │   └── TimeSeriesDatasetCreator.py         <- Creating final time series dataset
        ├── features
        │   └── feature_selection_an.py             <- Methods for anomaly feature selection
        ├── main.py
        └── models
            ├── AnomalyModel.py                     <- Anomaly training & Prediction
            ├── ClassificationModel.py              <- Misuse training & Prediction
            └── functions.py                        <- supporting funtions for models