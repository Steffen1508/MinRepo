# Skateboard Trick Detector

Miniprojekt i Deep Learning, AAU, efterår 2026.

Klassificerer fem skateboard-tricks fra video: Ollie, Shuvit, FrontShuvit, Kickflip og Heelflip. En R(2+1)D-18, pretrained på Kinetics-400, fine-tunes på et selvoptaget 120 fps-datasæt og sammenlignes med en ResNet-18, der kun ser enkeltbilleder. Som ekstra mulighed findes skateren med en Faster R-CNN, og klippet beskæres omkring skateren.

Hovedresultat: alle modeller rammer 0,83 til 1,00 macro F1 på et tilfældigt split, men frame-modellen klarer sig lige så godt som videomodellerne, og ingen af dem virker på optagelser af andre skatere. Først når der kommer varieret træningsdata med (SkateboardML), går den eksterne score fra 0,57 til 0,88.

## Filer

| Fil | Indhold |
|---|---|
| `dataset.py` | Finder klip og labels, finder trickket i tid via bevægelse, cache, PyTorch Dataset med augmentation |
| `detect.py` | Faster R-CNN (COCO) finder skater og skateboard, stabiliseret crop omkring skateren |
| `models.py` | `frame` (ResNet-18 per frame), `r2plus1d` og `r3d` med tre frysningsniveauer |
| `train.py` | Træning, early stopping, evaluering, confusion matrices |
| `predict.py` | Demo: video ind, video ud med bounding box, trick og confidence |
| `download_datasets.py` | Henter begge datasæt |
| `run_train.slurm` | Job-script til AI-LAB |
| `groups_place.csv` | Spot per klip, til gruppe-split (`--groups-csv`) |
| `results/` | Metrikker, træningslog, split, confusion matrices og forudsigelser for alle kørsler |

## Data

- **Træning:** [Skateboarding Trick Dataset (120fps)](https://www.kaggle.com/datasets/sensui0731/skateboarding-trick-dataset-120fps) af sensui0731, CC BY 4.0. Ligger i `Dataset_Kaggle/`.
- **Eksternt test:** [SkateboardML](https://github.com/LightningDrop/SkateboardML) af LightningDrop. Ligger i `SkateboardML-1.0/`. Repoets egne `trainlist02.txt` og `testlist02.txt` bruges som split.

Videoerne er ikke med i repoet. Hent dem med `python download_datasets.py` eller manuelt fra linkene.

## Opsætning

```bash
pip install -r requirements.txt
```

På AI-LAB bruges PyTorch-containeren med et lille venv til OpenCV, se kommentaren øverst i `run_train.slurm`.

## Kørsel

```bash
# Træning (lokalt eller med sbatch run_train.slurm <samme flag>)
python train.py --model frame --name baseline_frame
python train.py --model r2plus1d --freeze partial --name r21d_partial
python train.py --model r2plus1d --freeze partial --crop-mode skater \
    --splits-csv runs/r21d_partial/splits.csv --name r21d_partial_skater

# Med SkateboardML i træningen
python train.py --model r2plus1d --freeze partial --crop-mode skater \
    --splits-csv runs/r21d_partial/splits.csv \
    --extra-train-root SkateboardML-1.0/Tricks --extra-train-list SkateboardML-1.0/trainlist02.txt \
    --external-root SkateboardML-1.0/Tricks --external-list SkateboardML-1.0/testlist02.txt \
    --name r21d_partial_skater_extra

# Split efter optagelsessted
python train.py --model r2plus1d --freeze partial --groups-csv groups_place.csv --name r21d_partial_place

# Demo
python predict.py klip.mov --checkpoint runs/r21d_partial_skater/best.pt --out demo.mp4
```

## Resultater

Sensui-test: 40 klip, fem klasser. SkateboardML-test: 44 klip (15 ollies, 29 kickflips). "2-way F1" er macro F1, når modellen kun må vælge mellem ollie og kickflip. Altid at gætte kickflip giver 0,40.

| Kørsel | Sensui acc | Sensui F1 | SkateboardML 2-way F1 |
|---|---|---|---|
| Frame baseline | 0.95 | 0.95 | 0.41 |
| Frame baseline + crop | 1.00 | 1.00 | 0.42 |
| R(2+1)D linear probe | 0.50 | 0.47 | n/a |
| R(2+1)D partial | 0.83 | 0.83 | 0.46 |
| R(2+1)D full | 0.95 | 0.95 | 0.36 |
| R(2+1)D partial + crop | 0.93 | 0.93 | 0.57 |
| R(2+1)D partial + crop + SkateboardML train | 0.95 | 0.95 | 0.88 |
| Spot-split, R(2+1)D partial | 0.14 | 0.16 | 0.39 |
| Spot-split, R(2+1)D partial + crop | 0.13 | 0.14 | 0.47 |

Detaljer for hver kørsel ligger i `results/<kørsel>/`.

## Brug af AI

Koden er udviklet med hjælp fra Claude til insættelse af kommentare, rengøring af kode og slurm jobs til AI lab, co-pilot line suggestion er brugt til støresledelen af koden. Valg af problem, tricks og datasæt, kørsel af alle eksperimenter på AI-LAB og fortolkning af resultaterne er lavet af mig.
