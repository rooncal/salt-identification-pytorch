from dataloaders.custom_transformations import RandomRotate, RandomHorizontalFlip, RandomVerticalFlip, ToTensor, Resize, Normalize_Mask, Compose
from dataloaders.datasets.segmentation_dataset import DataSetSegmentation
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from utils.custom_loss_functions import DiceLoss, FocalLoss
from utils.evaluator import Evaluator
from modeling.unet import GenericUnet
from modeling.unet_resnet import UnetResnet
from mypath import Path


# trainer_unet = Trainer(
#                          experiment_name="Unet_Dice_Dropout_No_BN",
#                          model=GenericUnet,
#                          model_args= (3, 1, 0.2, False),
#                          loss_function=DiceLoss,
#                          optimizer=torch.optim.RMSprop,
#                          optimizer_kwargs={
#                                           "lr": 0.00005,
#                                         },
#                          lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
#                          lr_scheduler_kwargs={
#                                             "mode": "max",
#                                             "patience": 2,
#                          },
#                          train_dataloader=data_loader_train,
#                          val_dataloader=data_loader_val,
#                          model_checkpoint_dir="/content/drive/MyDrive/experiments/SIC/DualWriter/checkpoints",
#                          tensorboard_log_dir="/content/drive/MyDrive/experiments/SIC/DualWriter/runs",
#                          metrics_list=["mIoU"]
#                          )

class Translator(object):
    def __init__(self, args):
        self.args = args
        self.transformations = self.get_transformations()
        self.translated_args = {
            "experiment_name": args.experiment_name,
            "model": self.get_model(),
            "dataset": self.get_dataset(),
            "model_args": self.get_model_args(),
            "loss_function": self.get_loss_function(),
            "optimizer": self.get_optimizer(),
            "optimizer_kwargs": self.get_optimizer_kwargs(),
            "lr_scheduler": self.get_lr_scheduler(),
            "lr_scheduler_kwargs": self.get_lr_scheduler_args(),
            "train_dataloader": self.get_dataloader("train"),
            "val_dataloader": self.get_dataloader("val"),
            "model_checkpoint_dir": args.checkpoint_dir,
            "tensorboard_log_dir": args.tensorboard_dir,
            "metrics_list": self.get_metrics_list(),
            "tqdm": not args.no_tqdm,
            "start_epoch": args.start_epoch,
        }        
    
    def get_transformations(self):
        transform = Compose([
                Resize(),
                RandomRotate(),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                ToTensor(),
                Normalize_Mask(),
            ])
        return transform

    def get_model(self):
        if self.args.model == 'unet':
            return GenericUnet
        elif self.args.model == 'unet-resnet':
            return UnetResnet
        else:
            print("Model Undefined!")
            raise NotImplementedError

    def get_dataset(self):
        return Path.db_root_dir(self.args.dataset)
    
    def get_model_args(self):
        self.model_args = []
        if self.args.model == "unet":
            self.model_args += [3, 1, 0.2]
        elif self.args.model == "unet-resnet":
            self.model_args += [16, 0.2]
        if self.args.no_bn:
            self.model_args += [False]
        return self.model_args
    
    def get_loss_function(self):
        loss_functions = {
            "ce": nn.BCEWithLogitsLoss,
            "dice": DiceLoss,
            "focal": FocalLoss,
        }
        return loss_functions[self.args.loss_function]

    
    def get_optimizer(self):
        optimizers = {
            "adam": torch.optim.Adam,
            "RMSProp": torch.optim.RMSprop,
            "SGD": torch.optim.SGD,
        }
        return optimizers[self.args.optimizer]


    def get_optimizer_kwargs(self):
        optimizer_kwargs = {}
        optimizer_kwargs["lr"] = self.args.lr
        if self.args.optimizer == "SGD":
            if self.args.nesterov:
                optimizer_kwargs["nesterov"] = True
            optimizer_kwargs["momentum"] = self.args.momentum
        optimizer_kwargs["weight_decay"] = self.args.weight_decay
        return optimizer_kwargs

    def get_lr_scheduler(self):
        schedulers = {
            "reduce-on-plateau": nn.optim.lr_scheduler.ReduceLROnPlateau,
            "step": nn.optim.lr_scheduler.StepLR,
            "cos": nn.optim.lr_scheduler.CosineAnnealingLR,
        }
        return schedulers[self.args.lr_scheduler]
        
    def get_lr_scheduler_args(self):
        kwargs = {}
        if self.args.lr_scheduler == "reduce-on-plateau":
            kwargs["patience"] = self.args.patience
            kwargs["mode"] = "max"
        if self.args.lr_scheduler == "cos":
            kwargs["T_max"] = 10
        if self.args.lr_schduler == "step":
            kwargs["step_size"] = 10
        return kwargs

    def get_dataloader(self, phase):
        images_path, masks_path = Path.db_root_dir(args.dataset)
        dataset = DataSetSegmentation(images_path=images_path, masks_path=masks_path, transform=self.transformations)
        if phase == "train":
            return DataLoader(dataset,self.args.batch_size,sampler=sampler.SubsetRandomSampler(range(3500)))
        return DataLoader(dataset,self.args.batch_size,sampler=sampler.SubsetRandomSampler(range(3500,4000)))
    
    def get_metrics_list(self):
        if self.args.model in ["unet", "unet-resnet"]:
            return ["mIoU"]
        return ["Accuracy"]

        


