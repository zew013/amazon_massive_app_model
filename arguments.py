import argparse
import os

from transformers.optimization import SchedulerType

def params():
    parser = argparse.ArgumentParser()

    # Experiment options
    parser.add_argument("--task", default="baseline", type=str,\
                help="baseline is fine-tuning bert for classification;\n\
                      tune is advanced techiques to fine-tune bert;\n\
                      constast is contrastive learning method")
                      #choices=['baseline','tune','supcon'])
    parser.add_argument("--temperature", default=0.7, type=float, 
                help="temperature parameter for contrastive loss")

    # optional fine-tuning techiques parameters
    parser.add_argument("--reinit_n_layers", default=0, type=int, 
                help="number of layers that are reinitialized. Count from last to first.")
    
     # layer-wise lr, weight decay
    parser.add_argument("--lldr", default = False, type=bool,
                        help="layer-wise learning rate decay")
    parser.add_argument("--head-lr", default=0.00014 , type=float,
                help="learning rate head.")
    parser.add_argument("--init-lr", default=0.00013 , type=float,
                help="learning rate initial layer")
    
    parser.add_argument("--head-decay", default=0.01 , type=float,
                help="weight decay for head")
    parser.add_argument("--hidden-decay", default=0.01 , type=float,
                help="weight decay for hidden layers")
    
    
    parser.add_argument("--scheduler-type", default=SchedulerType.CONSTANT, type=SchedulerType,
                       choices=SchedulerType, help="scheduler type")
    
    parser.add_argument("--end-lr-ratio", default=1-1e6, type=float,
                        help="end lr percent")
    
    parser.add_argument("--warmup-steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--swa-start", default=10, type=int,
                        help="Stochastic Weight Averaging Start step")

    parser.add_argument("--swa-lr", default=2e-6, type=int,
                        help="Stochastic Weight Averaging Learning Rate")
    
    # Others
    parser.add_argument("--input-dir", default='assets', type=str, 
                help="The input training data file (a text file).")
    parser.add_argument("--output-dir", default='results', type=str,
                help="Output directory where the model predictions and checkpoints are written.")
    parser.add_argument("--model", default='bert', type=str,
                help="The model architecture to be trained or fine-tuned.")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--dataset", default="amazon", type=str,
                help="dataset", choices=['amazon'])
    

    # Key settings
    parser.add_argument("--ignore-cache", action="store_true",
                help="Whether to ignore cache and create a new input data")
    parser.add_argument("--debug", action="store_true",
                help="Whether to run in debug mode which is exponentially faster")
    parser.add_argument("--do-train", action="store_true",
                help="Whether to run training.")
    parser.add_argument("--do-eval", action="store_true",
                help="Whether to run eval on the dev set.")
    
    # Hyper-parameters for tuning
    parser.add_argument("--batch-size", default=256, type=int,
                help="Batch size per GPU/CPU for training and evaluation.")
    parser.add_argument("--learning-rate", default=0.0001, type=float,
                help="Model learning rate starting point.")
    parser.add_argument("--hidden-dim", default=10, type=int,
                help="Model hidden dimension.")
    parser.add_argument("--drop-rate", default=0.9, type=float,
                help="Dropout rate for model training")
    parser.add_argument("--embed-dim", default=10, type=int,
                help="The embedding dimension of pretrained LM.")
    parser.add_argument("--adam-epsilon", default=1e-8, type=float,
                help="Epsilon for Adam optimizer.")
    parser.add_argument("--n-epochs", default=10, type=int,
                help="Total number of training epochs to perform.")
    parser.add_argument("--max-len", default=20, type=int,
                help="maximum sequence length to look back")
    
    
    # extra parameters
    # lr_decay_step
    parser.add_argument("--lr-decay-step", default=1, type=int,
                        help="The step of learning rate decay.")
    parser.add_argument("--lr_decay_gamma", default=1, type=float,
                        help="The gamma of learning rate decay.")
    
    
    args = parser.parse_args()
    return args
