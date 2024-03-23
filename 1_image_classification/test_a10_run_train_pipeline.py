# Image Classification: Train pipeline
#   dataset: CIFAR10
#   models: GoogleNet, ResNet, ResNetPreAct
#   framework: PyTorch Lightning
#
#   https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial5/Inception_ResNet_DenseNet.html
#
#   Namgook Cho $ 02/24/24 $
#
import sys, time

from data import calculate_whitening, prep_loaders
from model import *
from googlenet import GoogleNet
from resnet import ResNet
from train import *
from utils import set_seed


def main():
    start = time.time()
    set_seed(42)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    #
    # 1) Set parameters ------------------------------------------------------------------------------------------------
    # download the pre-trained models
    # read_pretrained_models()


    #
    # 2) Prep Data-loaders ---------------------------------------------------------------------------------------------
    # 2-1) calculate mean & std of train-set
    data_mean, data_std = calculate_whitening()
    
    # 2-2) prepare data-loaders
    train_loader, val_loader, test_loader = prep_loaders(data_mean, data_std)


    #
    # 3) System training -----------------------------------------------------------------------------------------------
    # 3-1) it maps the model-name to the model-class
    # model_dict['GoogleNet'] = GoogleNet
    model_dict['ResNet'] = ResNet

    # 3-2) run the train-model
    if 0:
        googlenet_model, googlenet_results = train_model(model_name='GoogleNet',
                                                         train_loader=train_loader,
                                                         val_loader=val_loader,
                                                         test_loader=test_loader,
                                                         device=device,
                                                         model_hparams={'num_classes':10, 'act_fn_name':'relu'},
                                                         optimizer_name='Adam',
                                                         optimizer_hparams={'lr':1e-3, 'weight_decay':1e-4})
        print('GoogleNet results', googlenet_results)

    resnet_model, resnet_results = train_model(model_name='ResNet',
                                               train_loader=train_loader,
                                               val_loader=val_loader,
                                               test_loader=test_loader,
                                               device=device,
                                               model_hparams={'num_classes':10,
                                                              'c_hidden':[16,32,64],
                                                              'num_blocks':[3,3,3],
                                                              'act_fn_name':'relu'},
                                               optimizer_name='SGD',
                                               optimizer_hparams={'lr':0.1,
                                                                  'momentum':0.9,
                                                                  'weight_decay':1e-4})
    print('ResNet results', resnet_results)

    elapsed = time.time() - start
    print('Elapsed {:.2f} minutes'.format(elapsed/60.0))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError,IOError) as e:
        sys.exit(e)
