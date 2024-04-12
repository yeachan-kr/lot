import os
import sys
import argparse
import time
import datetime

# Custom Library
from utils import set_random_seed, load_seq2seq_dataset, Logger, get_num_classes


def run(args):
    # Create directories if not exist.
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    now = datetime.datetime.now()
    log_file_name = '{}_{}'.format(args.alg, now)
    print('Create LOG file {}'.format(log_file_name))
    sys.stdout = Logger(location=os.path.join(args.logdir, log_file_name))
    print(args)

    if not os.path.exists(args.modeldir):
        os.makedirs(args.modeldir)

    # Data partitioning based on non-iid strategy
    num_class = get_num_classes(args.dataset)
    seq2seq_dataset = load_seq2seq_dataset(dataset=args.dataset, model=args.model, datadir=args.datadir, num_class=num_class, max_length=args.max_seq_len)
    
    # Select Solver based on learning strategy
    solver = None
    if args.alg == 'full':
        from solvers.full_solver import FullSolver
        solver = FullSolver(args=args, dataset=seq2seq_dataset)      
    elif args.alg == 'lot':
        from solvers.lot_solver import LoTSolver
        solver = LoTSolver(args=args, dataset=seq2seq_dataset)    
    solver.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='trec', help='dataset used for training')
    parser.add_argument('--max_seq_len', type=int, default=128, help='')
    parser.add_argument('--warmup_epoch', type=int, default=1, help='')

    # Training configuration
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=1.0, help='learning rate (default: 5e-5)')
    parser.add_argument('--epochs', type=int, default=20, help='number of local epochs')
    parser.add_argument('--warmup_steps', type=int, default=0, help='')
    parser.add_argument('--reg_weight', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0.3)

    # Model configuration
    parser.add_argument('--alg', type=str, default='full', help='communication strategy: full/last')
    parser.add_argument('--model', type=str, default='prajjwal1/bert-tiny', help='neural network used in training (please refer to huggingface\'s model list')
    parser.add_argument('--adapter', type=str, default='none', help='')
    parser.add_argument('--rank', type=int, default=16, help='')
    parser.add_argument('--gradient_accumulation_step', type=int, default=1)

 
    # Directory configuration conda activate torch37
    parser.add_argument('--datadir', type=str, required=False, default="./data", help="Data directory")
    parser.add_argument('--logdir', type=str, default='./log/', help='dataset used for training')
    parser.add_argument('--modeldir', type=str, default='./save/', help='dataset used for training')
    parser.add_argument('--task_grad', type=str, default='.', help='path for token gradients')


    # Computation configuration
    parser.add_argument('--device', type=str, default='3', help='The device to run the program')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--logging_step', type=int, default=500, help="Random seed")

    args = parser.parse_args()


    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    set_random_seed(args.init_seed)

    # Start solver
    run(args)

