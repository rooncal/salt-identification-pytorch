import torch

class Evaluator(object):
    def __init__(self, writer, metrics_list=[], smooth=0):
      self.smooth = smooth
      self.writer = writer
      self.metrics_list = metrics_list
      self.mIoU =  0
      self.correct = 0
      self.accuracy = 0
      self.batch_count = 0
      self.num_samples = 0

    def _Intersection_over_Union(self, inputs, targets):

        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + self.smooth)/(union + self.smooth)

        return IoU

    def add_batch(self, output, mask):
      self.batch_count += 1
      if "mIoU" in self.metrics_list:
        self.mIoU = ((self.batch_count - 1) * self.mIoU + self._Intersection_over_Union(output, mask)) / self.batch_count
      if "Accuracy" in self.metrics_list:
        _, preds = output.max(1)
        self.correct += (preds == mask).sum() 
        self.num_samples += preds.size(0)
    
    def get_write_function(self, phase):
      if phase == "val":
        return self.writer.add_scalar_validation
      return self.writer.add_scalar_training

    def display_and_record(self, epoch, running_loss, phase, length):
      write = self.get_write_function(phase)
      print(f"Epoch: {epoch}, { 'Validation' if phase == 'val' else 'Training'} Loss: {running_loss / length :.4f}")
      write(f"Loss", running_loss / length, epoch)
      for metric in self.metrics_list:
        if metric == "mIoU":
          write('mIoU', self.mIoU, epoch)
          print(f"Epoch: {epoch}, mIoU: {self.mIoU}")
        if metric == "Accuracy":
          self.accuracy = float(self.correct) / self.num_samples
          write('Accuracy', self.accuracy, epoch)
          print(f"Epoch: {epoch}, Accuracy: {self.accuracy}")


    def reset_metric_accumulators(self):
      self.mIoU = 0
      self.correct = 0
      self.num_samples = 0
      self.accuracy = 0
      self.batch_count = 0
