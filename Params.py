import argparse


def ParseArgs():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch', default=1024, type=int, help='batch size')
    parser.add_argument('--tstBat', default=1024, type=int, help='number of users in a testing batch')
    parser.add_argument('--reg', default=1e-4, type=float, help='weight decay regularizer')#原1e-4
    parser.add_argument('--epoch', default=500, type=int, help='number of epochs')
    parser.add_argument('--latdim', default=64, type=int, help='embedding size')
    parser.add_argument('--gnn_layer', default=3, type=int, help='number of gnn layers')#3
    parser.add_argument('--topk10', default=10, type=int, help='K of top K')
    parser.add_argument('--topk20', default=20, type=int, help='K of top K')
    parser.add_argument('--topk40', default=40, type=int, help='K of top K')
    parser.add_argument('--data', default='yelp', type=str, help='name of dataset')
    parser.add_argument('--ssl_reg', default=0.1, type=float, help='weight for contrative learning')
    parser.add_argument("--ib_reg", type=float, default=0.1, help='weight for information bottleneck')
    parser.add_argument('--temp', default=0.3,type=float, help='temperature in contrastive learning')
    parser.add_argument('--tstEpoch', default=1, type=int, help='number of epoch to test while training')
    parser.add_argument('--gpu', default=1, type=int, help='indicates which gpu to use')
    parser.add_argument("--seed", type=int, default=421, help="random seed")
    parser.add_argument('--model_name', default='mine', type=str, help='the name of the model')
    parser.add_argument('--device', default='0', type=str, help='device_id')
    parser.add_argument('--save_name', type=str, default='tem')
    parser.add_argument('--interval', type=int, default=2)
    parser.add_argument('--a', type=float, default=0.2)
    parser.add_argument('--b', type=float, default=0.8)
    parser.add_argument('--mask', type=float, default=0.5)#lastfm 0.7 ciao 0.5 yelp 0.5
    parser.add_argument('--uu_layers', type=int, default=2)#2
    parser.add_argument('--ui_layers', type=int, default=3)#3
    parser.add_argument('--lambda1', type=float, default=0.1)
    parser.add_argument('--lambda2', type=float, default=1.0)
    parser.add_argument('--lambda3', type=float, default=1e-4)
     # params for the denoiser
    parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
    parser.add_argument('--dims', type=int, default=64, help='the dims for the DNN')
    parser.add_argument('--norm', type=bool, default=True, help='Normalize the input or not')
    parser.add_argument('--emb_size', type=int, default=16, help='timestep embedding size')

    # params for diffusions
    parser.add_argument('--steps', type=int, default=20, help='diffusion steps')
    parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
    parser.add_argument('--noise_scale', type=float, default=1, help='noise scale for noise generating')
    parser.add_argument('--noise_min', type=float, default=0.0001, help='noise lower bound for noise generating')
    parser.add_argument('--noise_max', type=float, default=0.01, help='noise upper bound for noise generating')
    parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
    parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
    parser.add_argument('--reweight', type=bool, default=True,
                        help='assign different weight to different timestep or not')
    parser.add_argument('--difflr', type=float, default=1e-3)

    args = parser.parse_args() 
    return args

args = ParseArgs()