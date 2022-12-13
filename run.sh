mkdir assets
#python main.py --n-epochs 8 --do-train --task baseline 
#python main.py --n-epochs 8 --do-train --task custom --reinit_n_layers 3
python main.py --n-epochs 10 --do-train --task custom --reinit_n_layers 3 --warmup-steps 50 --scheduler-type linear --swa-start 3
#python main.py --n-epochs 14 --do-train --task supcon --batch-size 64
