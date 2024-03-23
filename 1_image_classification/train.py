import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from model import CIFARModule

CHECKPOINT_PATH = '/home/ncho/Unzipped/0_prac/6_vtf/1_lightning/1_image_classification/saved_models'


def read_pretrained_models():
    import urllib.request
    from urllib.error import HTTPError
    
    base_url = 'https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial5/'
    pretrained_files = ['GoogleNet.ckpt', 'ResNet.ckpt', 'ResNetPreAct.ckpt', 'DenseNet.ckpt',
                        'tensorboards/GoogleNet/events.out.tfevents.googlenet',
                        'tensorboards/ResNet/events.out.tfevents.resnet',
                        'tensorboards/ResNetPreAct/events.out.tfevents.resnetpreact',
                        'tensorboards/DenseNet/events.out.tfevents.densenet']

    # create checkpoint path if it doesn't exist yet
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    # check whether it already exists for each file. If not, try downloading it.
    for file_name in pretrained_files:
        file_path = os.path.join(CHECKPOINT_PATH, file_name)
        if '/' in file_name:
            os.makedirs(file_path.rsplit('/',1)[0], exist_ok=True)
        if not os.path.isfile(file_path):
            file_url = base_url + file_name
            print(f'Downloading {file_url}...')
            try:
                urllib.request.urlretrieve(file_url, file_path)
            except HTTPError as e:
                print('Something went wrong. Please try to download the file from the GDrive folder\n', e)


def train_model(model_name, train_loader, val_loader, test_loader, device, save_name=None, **kwargs):
    '''
    Inputs:
        model_name - name of the model you want to run which is used to look up the class in 'model_dict'
        save_name (optional) - if specified, this name will be used for creating the checkpoint and logging directory
    '''
    if save_name is None:
        save_name = model_name

    # create a PyTorch Lightning trainer with the generation callback
    # default_root_dir: where to save models
    # accelerator: run on a GPU (if possible)
    # devices: how many GPUs/CPUs we want to use
    # max_epochs: how many epochs to train for if no patience is set
    # callbacks: save the best checkpoint based on the maximum val_acc recorded (only weights and not optimizer)
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),
                         accelerator='gpu' if str(device).startswith('cuda') else 'cpu',
                         devices=1,
                         max_epochs=180,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode='max', monitor='val_acc'),
                                    LearningRateMonitor('epoch')],
                         enable_progress_bar=True)
    trainer.logger._log_graph = True         # if True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None # optional logging argument that we don't need

    # check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, save_name + '.ckpt')
    if os.path.isfile(pretrained_filename):
        print(f'Found pretrained model at {pretrained_filename}, loading...')
        model = CIFARModule.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42) # reproducable
        model = CIFARModule(model_name=model_name, **kwargs)
        trainer.fit(model, train_loader, val_loader)
        # load best checkpoint after training
        model = CIFARModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # test best model on val-set and test-set
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {'test': test_result[0]['test_acc'], 'val': val_result[0]['test_acc']}
    return model, result
