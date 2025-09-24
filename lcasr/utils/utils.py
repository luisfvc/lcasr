import os
import yaml
import torch


def load_yaml(yaml_fn):
    with open(yaml_fn, 'rb') as f:
        content = yaml.load(f, Loader=yaml.FullLoader)
    return content


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def set_remote_paths(args, audio_pretrain=False, score_pretrain=False, mixed_pretrain=False):

    hostname = os.uname()[1]
    rks = ["rechenknecht%d.cp.jku.at" % i for i in range(9)] + ["rechenknecht%d" % i for i in range(9)]

    args.exp_root = args.remote_exp_root if hostname in rks else args.local_exp_root
    args.split_root = args.remote_split_root if hostname in rks else args.local_split_root

    # if hostname in rks:
    #     args.rir_root = '/share/cp/datasets/impulse_response_filters/mcdermottlab_22k'
    # else:
    #     args.rir_root = '/home/luis/dev/impulse_response_filters/resampled_rirs/mcdermottlab_22k'

    if audio_pretrain:
        args.maestro_root = args.remote_maestro_root if hostname in rks else args.local_maestro_root
        return args

    if score_pretrain:
        args.scores_root = args.remote_scores_root if hostname in rks else args.local_scores_root
        return args

    if mixed_pretrain:
        args.maestro_root = args.remote_maestro_root if hostname in rks else args.local_maestro_root
        args.scores_root = args.remote_scores_root if hostname in rks else args.local_scores_root
        return args

    args.msmd_root = args.remote_msmd_root if hostname in rks else args.local_msmd_root
    args.umc_root = args.remote_umc_root if hostname in rks else args.local_umc_root

    return args


def load_pretrained_model(args, network):
    model_dict = network.state_dict()

    cca_ref = '_est_UV' if args.refine_cca else ''
    params_dir = os.path.join(args.exp_root, 'trained_models', f'msmd_baseline{cca_ref}')
    pretrained_model_path = os.path.join(params_dir, f"params{cca_ref}.pt")
    # encoder = 'audio' if audio else 'score'
    # net = 'y' if audio else 'x'
    #
    # audio = True if args.finetune_mixed else audio
    # encoder = 'mixed' if args.finetune_mixed else encoder
    #
    # params_dir = os.path.join(args.exp_root, 'pretrained_models')
    # if args.ft_audio_path and audio:
    #     params_dir = args.ft_audio_path
    # if args.ft_score_path and not audio:
    #     params_dir = args.ft_score_path
    # pretrained_model_path = os.path.join(params_dir, f"params_{encoder}{f'_{args.audio_context}' if audio else''}.pt")
    # pretrained_model_path = pretrained_model_path.replace('.pt', '_lm.pt') if args.ft_last_run else pretrained_model_path

    pretrained_dict = torch.load(pretrained_model_path)['model_params']

    pretrained_x_net = {k.replace('x_net', 'x_net.cnn_encoder'): v for k, v in pretrained_dict.items() if
                        k.replace('x_net', 'x_net.cnn_encoder') in model_dict and 'x_net' in k}
    pretrained_y_net = {k.replace('y_net', 'y_net.cnn_encoder'): v for k, v in pretrained_dict.items() if
                        k.replace('y_net', 'y_net.cnn_encoder') in model_dict and 'y_net' in k}
    pretrained_y_cca = {k.replace('cca_layer', 'y_net.cca_layer'): v for k, v in pretrained_dict.items() if
                        k.replace('cca_layer', 'y_net.cca_layer') in model_dict and 'cca_layer' in k}
    pretrained_x_cca = {k.replace('cca_layer', 'x_net.cca_layer'): v for k, v in pretrained_dict.items() if
                        k.replace('cca_layer', 'x_net.cca_layer') in model_dict and 'cca_layer' in k}

    # overwrite entries in the existing state dict
    model_dict.update(pretrained_x_net)
    model_dict.update(pretrained_y_net)
    model_dict.update(pretrained_y_cca)
    model_dict.update(pretrained_x_cca)

    # load the new state dict
    network.load_state_dict(model_dict)
    print(f'Pretrained CNN encoders loaded from {pretrained_model_path}')

    if args.freeze_cnn:
        network.x_net.cnn_encoder.requires_grad_(False)
        network.y_net.cnn_encoder.requires_grad_(False)
        print('CNN encoders were frozen for training')

    return network