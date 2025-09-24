import argparse
import os
import random
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn
import wandb

from lcasr.utils.utils import set_remote_paths, make_dir, load_yaml
from lcasr.wrappers import create_snip_train_loaders, get_model
from lcasr.utils.metrics_tracker import initialize_metrics_summary, update_metrics_summary, print_epoch_stats
from lcasr.utils.train_utils import iterate_dataset
from lcasr.refine_cca import refine_cca


def train_bl_model(args, wandb_sweep=False, sweep_run_name=None):

    args = set_remote_paths(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ensuring reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

    # setting model paths and tag
    make_dir(args.exp_root)
    if not wandb_sweep:
        # training tag: trained, finetuned or pretrained
        train_tag = 'finetuned_models' if args.finetune else 'trained_models'
        train_tag = f"{train_tag}{f'/{args.run_name}' if args.separate_run else ''}"

        exp_path = args.exp_root
        if args.ft_path:
            exp_path = args.ft_path
        make_dir(os.path.join(exp_path, train_tag))

        # model tag
        model_tag = 'msmd_att' if args.use_att else 'msmd_baseline'
        model_tag_path = os.path.join(exp_path, train_tag, model_tag)
        make_dir(model_tag_path)

        # experiment tag
        dump_path = os.path.join(model_tag_path, f'params.pt')

    else:
        train_tag = 'wandb_sweeps'
        exp_path = args.exp_root
        make_dir(os.path.join(exp_path, train_tag))

        run_dir = os.path.join(exp_path, train_tag, sweep_run_name)
        make_dir(run_dir)

        dump_path = os.path.join(run_dir, f'params.pt')

    # getting dataloaders
    tr_loader, va_loader, tr_eval_loader = create_snip_train_loaders(args)

    # get model
    model, loss = get_model(args, snippet_model=True)

    # load pretrained models if finetuning
    # model = load_pretrained_model(args, model, audio=True) if (args.finetune_audio or args.finetune_mixed) else model
    # model = load_pretrained_model(args, model, audio=False) if args.finetune_score else model
    print(f'Saving model in {dump_path}')
    model.to(device)

    lr = args.lr
    if wandb_sweep:
        lr = float('{0:.0e}'.format(args.lr))
        print(f'Initial learning rate: {lr}')
    # get optimizer and scheduler
    # todo: create a wrapper function
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for _, vl in model.named_modules():
        if hasattr(vl, 'bias') and isinstance(vl.bias, nn.Parameter):
            pg2.append(vl.bias)  # biases
        if isinstance(vl, nn.BatchNorm2d) or isinstance(vl, nn.LayerNorm) or isinstance(vl, nn.GroupNorm):
            pg0.append(vl.weight)  # no decay
        elif hasattr(vl, 'weight') and isinstance(vl.weight, nn.Parameter):
            pg1.append(vl.weight)  # apply decay

    print(f"Using Adam with weight decay: {args.weight_decay}")
    optim = torch.optim.AdamW(pg0, lr=lr)

    optim.add_param_group({'params': pg1, 'weight_decay': args.weight_decay})  # add pg1 with weight_decay
    optim.add_param_group({'params': pg2})  # add pg2 (biases)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', factor=0.5, patience=args.patience,
                                                           verbose=True, min_lr=2e-6)

    # tracking training
    metrics_summary = initialize_metrics_summary()
    current_epoch_time = time.monotonic()

    for epoch in range(args.n_epochs):

        tr_loss = iterate_dataset(model, tr_loader, loss_function=loss, optimizer=optim, device=device, has_rnn=False)

        tr_metrics = iterate_dataset(model, tr_eval_loader, loss_function=loss, optimizer=None, device=device,
                                     has_rnn=False)
        tr_metrics['loss'] = tr_loss

        va_metrics = iterate_dataset(model, va_loader, loss_function=loss, optimizer=None, device=device, has_rnn=False)

        metrics_summary = update_metrics_summary(metrics_summary, epoch, tr_metrics, va_metrics)

        # check for improvement
        try:
            improvement = va_metrics['map'] >= max(metrics_summary['va_map'][:-1])
        except ValueError:
            improvement = True

        if improvement:
            best_model = model.state_dict()
            best_optim = optim.state_dict()

            if not args.no_dump_model:
                best_checkpoint = {'epoch': epoch, 'model_params': best_model, 'optim_state': best_optim}
                torch.save(best_checkpoint, dump_path)

        print_epoch_stats(metrics_summary, current_epoch_time, patience=args.patience - scheduler.num_bad_epochs)
        current_epoch_time = time.monotonic()
        if args.use_wandb or wandb_sweep:
            wandb_metrics = {metric: value[-1] for metric, value in metrics_summary.items()}
            wandb_metrics['va_map_max'] = max(metrics_summary['va_map'])
            wandb.log(wandb_metrics)
            if epoch > 30 and max(metrics_summary['va_map']) < 10:
                break

        scheduler.step(va_metrics['map'])

    if args.refine_cca:
        refine_cca(args, wandb_sweep=wandb_sweep, sweep_run_name=sweep_run_name)

    # if args.eval_all:
    #
    #     sh_args = ['python', 'eval_snippet_retrieval.py', '--ret_dir', 'both', '--refine_cca', '--dump_results']
    #     sh_args.append('--finetune_audio') if args.finetune_audio else None
    #     sh_args.append('--finetune_score') if args.finetune_score else None
    #     sh_args.append('--finetune_mixed') if args.finetune_mixed else None
    #     sh_args.append('--ft_last_run') if args.ft_last_run else None
    #     sh_args.extend(['--exp_path', exp_path])
    #     for dataset in ['MSMD', 'RealScores_Synth', 'RealScores_Rec']:
    #         ret = subprocess.call(sh_args + ['--dataset', dataset], shell=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--aug_config', help='augmentation configuration', type=str, default='full_aug')
    parser.add_argument('--use_att', help='use attention layer', action='store_true', default=False)
    parser.add_argument('--finetune', help='load pretrained encoder', action='store_true', default=False)
    parser.add_argument('--no_dump_model', help='save best model every epoch', action='store_true', default=False)
    parser.add_argument('--eval_all', help='evaluate snippet retrieval after train', action='store_true', default=False)
    parser.add_argument('--refine_cca', help='refine cca layer after training', action='store_true', default=False)
    parser.add_argument('--ft_path', help='location to save the finetuned models', type=str, default=None)
    parser.add_argument('--use_wandb', help='monitor training with wandb', action='store_true', default=False)
    parser.add_argument('--run_name', help='wandb run name', type=str, default='')
    parser.add_argument('--separate_run', help='save run separately according to run_name', action='store_true',
                        default=False)

    configs = load_yaml('config/msmd_snip_config.yaml')

    if parser.parse_args().use_wandb:
        wandb.init(config=configs, project='lcasr')
        if parser.parse_args().run_name:
            wandb.run.name = parser.parse_args().run_name

    for k, v in configs.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    train_bl_model(parser.parse_args())
