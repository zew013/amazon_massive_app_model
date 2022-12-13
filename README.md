# Intention Classification with Pretrained Language Models

This respository is refered and modified from this repository of the [paper](https://arxiv.org/abs/2109.03079).

<br>

## Prerequisites
To run our model, you need to install tqdm, transformers, datasets, UMAP packages and their dependency packages.


## Help
You can use the following statement to see the detailed configuration of the parameters.
```
python main.py -h
```

## Running
For the training of all the models, you can pass the following command with appropriate paramters.
```
main.py [-h] 
        [--task TASK] 
        [--temperature TEMPERATURE] [--reinit_n_layers REINIT_N_LAYERS] [--lldr LLDR]
        [--head-lr HEAD_LR] 
        [--init-lr INIT_LR] 
        [--head-decay HEAD_DECAY] 
        [--hidden-decay HIDDEN_DECAY]
        [--scheduler-type {SchedulerType.LINEAR,SchedulerType.COSINE,SchedulerType.COSINE_WITH_RESTARTS,SchedulerType.POLYNOMIAL,SchedulerType.CONSTANT,SchedulerType.CONSTANT_WITH_WARMUP}]
        [--end-lr-ratio END_LR_RATIO] [--warmup-steps WARMUP_STEPS] 
        [--input-dir INPUT_DIR]
        [--output-dir OUTPUT_DIR] 
        [--model MODEL] 
        [--seed SEED] 
        [--dataset {amazon}] 
        [--ignore-cache] [--debug]
        [--do-train] 
        [--do-eval] 
        [--batch-size BATCH_SIZE] 
        [--learning-rate LEARNING_RATE]
        [--hidden-dim HIDDEN_DIM] 
        [--drop-rate DROP_RATE] 
        [--embed-dim EMBED_DIM] 
        [--adam-epsilon ADAM_EPSILON]
        [--n-epochs N_EPOCHS] 
        [--max-len MAX_LEN] [--lr-decay-step LR_DECAY_STEP]
        [--lr_decay_gamma LR_DECAY_GAMMA]
```
<br>

**(1) Baseline Model**
```
python main.py --n-epochs 10 --do-train --task baseline --temperature 0.4 --reinit_n_layers=0 --scheduler-type='linear' --end-lr-ratio=-999999.0 --warmup-steps=45 --input-dir='assets' --output-dir='results' --model='bert' --seed=42 --dataset='amazon' --batch-size=256 --learning-rate=0.0005 --hidden-dim=10 --drop-rate=0.5 --embed-dim=10 --adam-epsilon=1e-08 --lr-decay-step=1 --lr_decay_gamma=1
```


**(2) Custom Fine Tuning Model1 - LLDR**
```
python main.py --n-epochs 10 --do-train --task custom --temperature 0.4 --reinit_n_layers=0 --scheduler-type constant --end-lr-ratio=-999999.0 --warmup-steps=45 --input-dir='assets' --output-dir='results' --model='bert' --seed=42 --dataset='amazon' --batch-size=256 --learning-rate=0.0005 --hidden-dim=10 --drop-rate=0.5 --embed-dim=10 --adam-epsilon=1e-08 --max-len=20 --lr-decay-step=1 --lr_decay_gamma=1
```

**(3) Custom Fine Tuning Model - SWA**
```
python main.py --n-epochs 15 --do-train --task custom --temperature 0.4 --reinit_n_layers=0 --scheduler-type constant --end-lr-ratio=-999999.0 --warmup-steps=45 --input-dir='assets' --output-dir='results' --model='bert' --seed=42 --dataset='amazon' --batch-size=256 --learning-rate=0.0005 --hidden-dim=10 --drop-rate=0.5 --embed-dim=10 --adam-epsilon=1e-08 --max-len=20 --lr-decay-step=1 --lr_decay_gamma=1 --swa-start=6
```

**(4) Custom Fine Tuning Model - BOTH**
```
python main.py --n-epochs 15 --do-train --task custom --temperature 0.4 --reinit_n_layers=0 --scheduler-type constant --end-lr-ratio=-999999.0 --warmup-steps=45 --input-dir='assets' --output-dir='results' --model='bert' --seed=42 --dataset='amazon' --batch-size=256 --learning-rate=0.0005 --hidden-dim=10 --drop-rate=0.5 --embed-dim=10 --adam-epsilon=1e-08 --max-len=20 --lr-decay-step=1 --lr_decay_gamma=1 --lldr=true
```

**(5) SimCLR model**
```
python main.py --n-epochs 5 --do-train --task simclr --temperature 0.05 --reinit_n_layers 0  --scheduler-type constant --model='bert' --seed=42 --dataset='amazon' --batch-size=64 --learning-rate=0.00001 --hidden-dim=15 --drop-rate 0.1 --embed-dim 10 --adam-epsilon=1e-08 --max-len=20
```

**(6) SupCon model**
```
python main.py --n-epochs 10 --do-train --task supcon --temperature 0.05 --reinit_n_layers=0  --scheduler-type constant --model bert --seed=42 --dataset='amazon' --batch-size 256 --learning-rate=0.00001 --hidden-dim=15 --drop-rate=0.1 --embed-dim=10 --adam-epsilon=1e-08 --max-len=20
```
