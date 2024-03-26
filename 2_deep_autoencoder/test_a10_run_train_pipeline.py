# Autoencoder: Train pipeline
#   dataset: CIFAR10
#   latent dimensions: 64, 128, 256, 384
#   framework: PyTorch Lightning
#
#   https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html
#
#   Namgook Cho $ 03/26/24 $
#
import sys, time

import matplotlib.pyplot as plt
import pytorch_lightning as pl

from data import prep_loaders
from train import read_pretrained_models, train_cifar


def main():
    start = time.time()
    pl.seed_everything(42)


    #
    # 1) Set parameters ------------------------------------------------------------------------------------------------
    # download the pre-trained models
    # read_pretrained_models()

    #
    # 2) Prep Data-loaders ---------------------------------------------------------------------------------------------
    train_loader, val_loader, test_loader = prep_loaders()

    #
    # 3) Train the model -----------------------------------------------------------------------------------------------
    model_dict = {}
    for latent_dim in [64, 128, 256, 384]:
        model_ld, result_ld = train_cifar(train_loader, val_loader, test_loader, latent_dim)
        model_dict[latent_dim] = {'model':model_ld, 'result':result_ld}

    #
    # 4) Plot the reconstruction error ---------------------------------------------------------------------------------
    latent_dims = sorted([k for k in model_dict])
    val_scores = [model_dict[k]['result']['val'][0]['test_loss'] for k in latent_dims]

    fig = plt.figure(figsize=(6, 4))
    plt.plot(latent_dims, val_scores, '--', color='#000', marker='*', markeredgecolor='#000', markerfacecolor='y',
             markersize=16)
    plt.xscale('log')
    plt.xticks(latent_dims, labels=latent_dims)
    plt.title('Reconstruction error over latent dimensionality', fontsize=14)
    plt.xlabel('Latent dimensionality')
    plt.ylabel('Reconstruction error')
    plt.minorticks_off()
    plt.ylim(0, 100)
    plt.show()

    elapsed = time.time() - start
    print('Elapsed {:.2f} minutes'.format(elapsed/60.0))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, IOError) as e:
        sys.exit(e)
