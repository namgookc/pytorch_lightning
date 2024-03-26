import os
import torch, torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from data import get_train_images
from aed import Autoencoder


CHECKPOINT_PATH = '/home/ncho/Unzipped/0_prac/6_vtf/1_lightning/2_deep_autoencoder/saved_models'

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def read_pretrained_models():
    import urllib.request
    from urllib.error import HTTPError

    base_url = 'https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial9/'
    pretrained_files = ['cifar10_64.ckpt', 'cifar10_128.ckpt', 'cifar10_256.ckpt', 'cifar10_384.ckpt']

    # create checkpoint path if it doesn't exist yet
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    # check whether it already exists for each file. If not, try downloading it
    for file_name in pretrained_files:
        file_path = os.path.join(CHECKPOINT_PATH, file_name)
        if not os.path.isfile(file_path):
            file_url = base_url + file_name
            print(f'Downloading {file_url}...')
            try:
                urllib.request.urlretrieve(file_url, file_path)
            except HTTPError as e:
                print('Something went wrong. Please try to download the file from the GDrive folder\n',e)


class GenerateCallback(pl.Callback):

    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = input_imgs # Images to reconstruct during training
        self.every_n_epochs = every_n_epochs # Only save those images every N epochs (otherwise tensorboard gets quite large)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(input_imgs)
                pl_module.train()
            # Plot and add to tensorboard
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0,1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, range=(-1,1))
            trainer.logger.experiment.add_image('Reconstructions', grid, global_step=trainer.global_step)


def train_cifar(train_loader, val_loader, test_loader, latent_dim):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, f'cifar10_{latent_dim}'),
                         accelerator='gpu' if str(device).startswith('cuda') else 'cpu',
                         devices=1,
                         max_epochs=500,
                         callbacks=[ModelCheckpoint(save_weights_only=True),
                                    GenerateCallback(get_train_images(8), every_n_epochs=10),
                                    LearningRateMonitor('epoch')])
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f'cifar10_{latent_dim}.ckpt')
    if os.path.isfile(pretrained_filename):
        print('Found pretrained model, loading...')
        model = Autoencoder.load_from_checkpoint(pretrained_filename)
    else:
        model = Autoencoder(base_channel_size=32, latent_dim=latent_dim)
        trainer.fit(model, train_loader, val_loader)
    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {'test': test_result, 'val': val_result}
    return model, result
