
### setup
```bash

conda create --name yolo1 python=3.9
conda activate yolo1
pip install -r requirements.txt

```

### check dataset
```bash

python inspect_dataset.py
python inspect_dataset.py --set val

```

### run
```bash

python dataset_generator.py

python train_simple.py
python train_simple.py --config config.yaml

python train.py --config config_optimized.yaml
python train.py --config config_optimized.yaml --resume runs/optimized_run/weights/checkpoint_epoch_12.pth

tensorboard --logdir runs

python inference_yolo1d.py --model best_model_peaks.pth --signal-type anomaly --iou-thresh=0.001
python inference_yolo1d.py --model checkpoint_improved_epoch_20.pth --signal-type anomaly --iou-thresh=0.001

python inference_yolo1d.py          \
    --model best_model_peaks.pth    \
    --signal-type anomaly           \
    --conf-thresh=0.5               \
    --iou-thresh=0.8888            \
    --signal-length=256

python inference_yolo1d.py          \
    --signal-type simple            \
    --conf-thresh=0.6               \
    --iou-thresh=0.00001            \
    --signal-length=1024            \
    --model best_model.pth

python inference_yolo1d.py --model best_model.pth

```
