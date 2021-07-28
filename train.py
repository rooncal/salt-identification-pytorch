import torch
import os
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.utils.tensorboard import SummaryWriter

from utils.evaluator import Evaluator
from translator import Translator


class DualSummaryWriter(object):
  def __init__(self, location, *args, **kwargs):
    location_training = os.path.join(location,"train")
    location_validation = os.path.join(location,"val")
    self.writer1 = SummaryWriter(location_training, *args, **kwargs) 
    self.writer2 = SummaryWriter(location_validation, *args, **kwargs) 

  def add_scalar_training(self,*args, **kwargs):
    self.writer1.add_scalar(*args, **kwargs)
  
  def add_scalar_validation(self,*args, **kwargs):
    self.writer2.add_scalar(*args, **kwargs)
  
  def close(self):
    self.writer1.close()
    self.writer2.close()

class Trainer():
  def __init__(self,
               model,
               loss_function,
               optimizer,
               train_dataloader,
               model_args=(),
               optimizer_kwargs={},
               start_epoch=0,
               val_dataloader=None,
               tqdm=True,
               tensorboard_log_dir='runs',
               lr_scheduler=None,
               lr_scheduler_kwargs={},
               model_checkpoint_dir='',
               experiment_name='model_training',
               metrics_list=[]
               ):
    self.model = model(*model_args)
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.model.to(device=self.device)
    self.loss_function = loss_function()
    self.epoch = start_epoch
    self.optimizer = optimizer(self.model.parameters(), **optimizer_kwargs)
    self.train_dataloader = train_dataloader
    self.val_dataloader = val_dataloader
    self.tensorboard_log_dir = tensorboard_log_dir
    self.model_checkpoint_dir = model_checkpoint_dir
    self.tqdm = tqdm
    self.experiment_name = f"{experiment_name}_{self.optimizer.defaults['lr']}"
    self.writer = DualSummaryWriter(f'{self.tensorboard_log_dir}/{self.experiment_name}')
    self.train_dataloader_length = len(train_dataloader)
    self.val_dataloader_length = len(val_dataloader)
    self.evaluator = Evaluator(self.writer, metrics_list=metrics_list)
    if lr_scheduler:
      self.lr_scheduler = lr_scheduler(self.optimizer, **lr_scheduler_kwargs)
    else:
      self.lr_scheduler = None

  def train(self, total_epochs, checkpoint_interval=10):
    for self.epoch in range(self.epoch, total_epochs):
      self.train_iter()
      if self.val_dataloader:
        self.val_iter()
      if (self.epoch + 1) % checkpoint_interval == 0:
        self.save_checkpoint()


  def train_iter(self):
    running_loss = 0
    self.model.train()
    self.evaluator.reset_metric_accumulators()

    if self.tqdm:
      train_dataloader = tqdm(self.train_dataloader, position=0, leave=True)

    for (image, mask) in train_dataloader:
      self.optimizer.zero_grad()
      image, mask = image.to(self.device), mask.to(self.device)
      output = self.model(image)
      loss = self.loss_function(output,mask)
      loss.backward()
      self.optimizer.step()
      running_loss += loss.item()
      self.evaluator.add_batch(output, mask)

    self.evaluator.display_and_record(self.epoch, running_loss, 'train', self.train_dataloader_length)

  def val_iter(self):
    self.model.eval()
    self.evaluator.reset_metric_accumulators()

    if self.tqdm:
      val_dataloader = tqdm(self.val_dataloader, position=0, leave=True)

    with torch.no_grad():
      running_loss = 0
      for (image, mask) in val_dataloader:
        image, mask = image.to(self.device), mask.to(self.device)
        output = self.model(image)
        loss = self.loss_function(output, mask)
        running_loss += loss.item()
        self.evaluator.add_batch(output, mask)

      self.evaluator.display_and_record(self.epoch, running_loss, 'val', self.val_dataloader_length)
      if self.lr_scheduler:
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
          self.lr_scheduler.step(self.evaluator.mIoU)
        else:
          self.lr_scheduler.step()
  
  def save_checkpoint(self):
    dirtree = os.path.join(self.model_checkpoint_dir,self.experiment_name)
    if not os.path.isdir(dirtree):
      os.makedirs(dirtree, exist_ok=True)
    torch.save({
          'epoch': self.epoch,
          'model_state_dict': self.model.state_dict(),
          'optimizer_state_dict': self.optimizer.state_dict(),
          'scheduler': self.lr_scheduler,
          }, os.path.join(self.model_checkpoint_dir,self.experiment_name,f"checkpoint_{self.epoch}")) 
      
  def load_checkpoint(self, path=None, epoch=0):
    if epoch and not path:
      path = os.path.join(self.model_checkpoint_dir,self.experiment_name,f"checkpoint_{epoch}")
    checkpoint = torch.load(path)
    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    self.epoch = checkpoint['epoch']
    self.lr_scheduler = checkpoint['scheduler']
    
    
def main():
  parser = argparse.ArgumentParser(description="PyTorch Unet/UnetResnet Training")
  parser.add_argument('--model', type=str, default='unet', choices=['unet','unet-resnet'],
                      help='model name (default: unet')
  parser.add_argument('--dataset', type=str, default='tgs_salt_identification',
                      choices=['tgs_salt_identification'], help='dataset name (default: tgs_salt_identification)')
  parser.add_argument('--no-bn', action='store_true', default=False,
                      help='will not use batch normalization (default: False')
  parser.add_argument('--loss-function', type=str, default='ce',
                      choices=['ce','dice','focal'],
                      help='loss function type (default: ce)')
  parser.add_argument('--epochs', type=int, default=10, metavar='N',
                      help='number of epochs to train (default: 10)')
  parser.add_argument('--start_epoch', type=int, default=0, metavar='N', 
                      help='starting epoch (default: 0)')
  parser.add_argument('--batch-size', type=int, default=None,
                      metavar='N', help='batch size for training (default: auto)')
  parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate (default: 0.00001)')
  parser.add_argument('--lr-scheduler', type=str, default='reduce-on-plateau',
                        choices=['reduce-on-plateau', 'step', 'cos'],
                        help='lr scheduler mode: (default: reduce-on-plateau)')
  parser.add_argument('--momentum', type=float, default=0.,
                        metavar='M', help='momentum (default: 0.9)')
  parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
  parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
  parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
  parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
  parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
  parser.add_argument('--resume-by-epoch', type=int, default=None,
                        help='put the epoch of the resuming file.')
  parser.add_argument('--checkpoint-interval', type=int, default=10,
                      help="checkpoint interval (default 10)")
  parser.add_argument('--no-tdqm', action='store_true', default=False,
                      help="disables tqdm while training")
  parser.add_argument('--optimizer', type=str, default="adam", choices=["adam", "RMSProp", "SGD"],
                      help="optimizer type (default: adam)")
  parser.add_argument('--patience', type=int, default=3, 
                      help="patience for on plateau lr scheduler")
  parser.add_argument('--experiment-name', type=str ,default="training",
                      help="experiment name, used for checkpoint and logs (default: training)")
  parser.add_argument('--tensorboard-dir', type=str, default="/content/drive/MyDrive/experiments/SIC/DualWriter/runs",
                      help='where to save tensorboard logs (default: /content/drive/MyDrive/experiments/SIC/DualWriter/runs )')
  parser.add_argument('--checkpoint-dir', type=str, default="/content/drive/MyDrive/experiments/SIC/DualWriter/checkpoints",
                      help="where to save the best checkpoint (default: content/drive/MyDrive/experiments/SIC/DualWriter/checkpoints)")

  args = parser.parse_args()
  args.cuda = not args.no_cuda and torch.cuda.is_available()

  if args.cuda:
    try:
        args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
    except ValueError:
        raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
  
  if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)
   
  translator = Translator(args)
  trainer = Trainer(**translator.translated_args)
  print("Starting epochs:", trainer.epoch)
  print("Total epochs:", args.epochs)
  trainer.train(args.epochs)
  trainer.writer.close()

  
if __name__ == "__main__":
  main()
