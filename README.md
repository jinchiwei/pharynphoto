# pharynphoto
classifying presence of pharyngitis using mobile phone capture and deep learning

```
pharyngitis_classification/
├── data/                     # web-sourced dataset*
├── src/
│   ├── dataloader.py         # dataset loading and transformations
│   ├── model.py              # model architectures
│   ├── train.py              # training and validation functions
│   └── main.py               # main script
├── results/
│   ├── dataloader.py         # dataset loading and transformations
│   ├── *.png                 # train/val accuracy/loss, confusion, heatmaps, roc
│   ├── test_accuracy.txt     # per-image class probabilities
│   └── training_results.csv  # per-epoch train results
├── environment.yml           # dependencies
└── README.md                 # documentation
```

*dataset courtesy of [yoo (2020)](https://data.mendeley.com/datasets/)
