import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import argparse

from torch.backends import cudnn
from utils.utils import *

from solver_baseline import Solver
txtDir_base = '/home/lixinying/anomaly_transformer/code/results/'

def str2bool(v):
    return v.lower() in ('true')


def main(config):
    config.subject_id = 1
    cudnn.benchmark = True
    # if (not os.path.exists(config.model_save_path)):
    #     mkdir(config.model_save_path)
    solver = Solver(vars(config))
    # if (not os.path.exists(config.model_save_path)):
    solver.train()
    for i in range(1,24):
        # config.subject_id = i
        print(str(i+1).rjust(2,'0'))
        txtDir = config.results_save_path #txtDir_base + config.model_name + '_test_results_pot_freq_0924.txt'
        with open(txtDir,'a+') as f:
            f.write(str(i+1).rjust(2,'0')+'\t')
        # cudnn.benchmark = True
        # if (not os.path.exists(config.model_save_path)):
        #     mkdir(config.model_save_path)
        # solver = Solver(vars(config))

        # if config.mode == 'train':
        #     solver.train()
        # elif config.mode == 'test':
        solver.test(sub_id=i)

        # return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--use_pot', type=bool, default=True)
    parser.add_argument('--win_size', type=int, default=256)
    parser.add_argument('--input_c', type=int, default=18)
    parser.add_argument('--output_c', type=int, default=18)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--step', type=int, default=32)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='CHBMIT')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='/data/lixinying/chb-mit_18ch/')
    parser.add_argument('--results_save_path', type=str, default='/home/lixinying/anomaly_transformer/code/results/COUTA_dsPOT_1119.txt')
    parser.add_argument('--model_save_path', type=str, default='/home/lixinying/anomaly_transformer/code/checkpoints/COUTA_1115.pth')
    parser.add_argument('--anormly_ratio', type=float, default=0.5)
    parser.add_argument('--model_name', type=str, default='COUTA', 
                        choices=['anomaly_transformer', 'anomaly_transformer_freq','COUTA','DCdetector','DIF', \
                            'NCAD','TranAD','TcnED','MSCRED','STEN','GDN'])

    config = parser.parse_args()

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)
