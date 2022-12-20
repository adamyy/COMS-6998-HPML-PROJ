import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import argparse
import sys
import time
import os
import copy
from sklearn.metrics import accuracy_score
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def set_cudnn_autotuner(enabled=False):
  cudnn.benchmark = enabled

def get_pretrained_model(use_torchscript=False):
  """get the pretrained ResNet50 model with its last FC layer replaced"""
  net = torchvision.models.resnet50(pretrained=True)
  net.fc = nn.Linear(2048, 100) # replace last linear layer with a new one
  net = net.to(device)
  if use_torchscript:
    net = torch.jit.trace(net, torch.rand(128, 3, 32, 32).to(device))
  return net

def get_dataloader(batch_size=128, num_workers=2, pin_memory=False):
  """get the training and validation dataloader"""
  train_data = torchvision.datasets.CIFAR100(download=True, root="./data", transform=data_transforms["train"])
  val_data = torchvision.datasets.CIFAR100(root="./data", train=False, transform=data_transforms["val"])
  train_dl = DataLoader(train_data, batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
  val_dl = DataLoader(val_data, batch_size, num_workers=num_workers, pin_memory=pin_memory)
  return train_dl, val_dl

def get_criterion():
  """get a new instance of the loss function"""
  criterion = nn.CrossEntropyLoss()
  return criterion

def get_optimizer(model):
  """get a new instance of the optimizer"""
  optimizer = optim.Adam(model.parameters())
  return optimizer

def train_model(model, criterion, optimizer, train_dl, val_dl, scheduler=None, n_epochs=20):
  """returns trained model"""
  for epoch in range(1, n_epochs + 1):
    train_losses = []
    val_losses = []
    # train the model #
    model.train()
    for batch_idx, (imgs, labels) in tqdm(enumerate(train_dl)):
      imgs = imgs.to(device)
      labels = labels.to(device)
      optimizer.zero_grad()
      output = model(imgs)
      # calculate loss
      loss = criterion(output, labels)
      # back prop
      loss.backward()
      # grad
      optimizer.step()
      train_losses.append(loss.item())

    # # validate the model #
    # model.eval()
    # for batch_idx, (imgs, labels) in enumerate(val_dl):
    #     imgs = imgs.to(device)
    #     labels = labels.to(device)
    #     output = model(imgs)
    #     # calculate loss
    #     loss = criterion(output, labels)
    #     val_losses.append(loss.item())
    
    # # print training/validation statistics 
    # print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
    #     epoch, np.mean(train_losses), np.mean(val_losses)))

    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, np.mean(train_losses)))

    if scheduler:
      scheduler.step()
  return model

def args_to_str(args):
  """Convert the training arguments to a slug string (for filenames)"""
  return "_".join([
      f"epochs={args.n_epochs}",
      f"batch={args.batch_size}",
      f"workers={args.num_workers}",
      f"pinmem={args.pin_memory}",
      f"autotune={args.cudnn_autotuner}",
      f"torchscript={args.use_torchscript}"
  ])

def parse_arguments(arguments=None):
  parser = argparse.ArgumentParser('resnet50-cifar100')
  parser.add_argument('--cudnn-autotuner', action='store_true',
                      help='Apply cuDNN autotuner.')
  parser.add_argument('--use-torchscript', action='store_true',
                      help='Use TorchScript JIT.')
  parser.add_argument('--batch-size', type=int, default=128)
  parser.add_argument('--num-workers', type=int, default=2)
  parser.add_argument('--pin-memory', action='store_true')
  parser.add_argument('--n-epochs', type=int, default=20)
  parser.add_argument('--profile', choices=['cprofile', 'torch'], default='cprofile')
  parser.add_argument('--dry-run', action='store_true')
  if arguments:
    return parser.parse_args(arguments)
  else:
    return parser.parse_args()

def evaluate_model(net, val_dl):
  """Return the model's accuracy on the validation set"""
  net.eval()
  preds = torch.Tensor([]).to(device)
  labels = torch.Tensor([]).to(device)
  for _, (imgs, batch_labels) in tqdm(enumerate(val_dl)):
    output = net(imgs.to(device))
    batch_preds = torch.argmax(output, dim=1)
    preds = torch.hstack([preds, batch_preds.to(device)])
    labels = torch.hstack([labels, batch_labels.to(device)])
  return accuracy_score(preds.cpu().detach().numpy(), labels.cpu().detach().numpy())


def main(arguments=None):
  os.makedirs("./model", exist_ok=True)
  os.makedirs("./profile", exist_ok=True)
  args = parse_arguments(arguments) # e.g., ["--n-epochs", "20"]
  print(args_to_str(args))
  if args.dry_run:
    return
  set_cudnn_autotuner(args.cudnn_autotuner)
  net = get_pretrained_model(args.use_torchscript)
  train_dl, val_dl = get_dataloader(args.batch_size, args.num_workers, args.pin_memory)
  criterion = get_criterion()
  optimizer = get_optimizer(net)

  # warm up
  train_model(net, criterion, optimizer, train_dl, val_dl, n_epochs=1)

  if args.profile == 'cprofile':
    import cProfile
    pr = cProfile.Profile()
    pr.enable()
    train_model(net, criterion, optimizer, train_dl, val_dl, n_epochs=args.n_epochs)
    pr.disable()
    pr.dump_stats(f"./profile/train_model_{args_to_str(args)}.profile")
    # cProfile.runctx('train_model(net, criterion, optimizer, train_dl, val_dl, n_epochs=args.n_epochs)',
    #                 globals(), locals(),
    #                 filename=f"train_model_{args_to_str(args)}.profile", sort='tottime')
    torch.save(net.state_dict(), f"./model/model_{args_to_str(args)}.pt")
    print(f"validation set accuracy: {evaluate_model(net, val_dl)}")
  elif args.profile == 'torch':
    from torch.profiler import profile, record_function, ProfilerActivity

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 profile_memory=True, record_shapes=True) as prof:
      train_model(net, criterion, optimizer, train_dl, val_dl, n_epochs=1) # force n_epochs to 1 to prevent OOM

if __name__ == '__main__':
  main()
