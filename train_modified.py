
import torch
print(torch.__version__)
# print(torch.cuda.is_available())
from collections import OrderedDict
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn 
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim
from torchvision import datasets, transforms, models
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

def parse_args():
  parser = argparse.ArgumentParser(description="Train a neural network on a flower dataset.")
  parser.add_argument("--data_dir", type=str,default="flowers", help="Path to the directory containing the flower dataset")
  parser.add_argument("--learning_rate", type=float, default=0.02, help="Learning rate for the optimizer during training")
  parser.add_argument("--hidden_units", type=int, default=550, help="Number of hidden units in the neural network")
  parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
  parser.add_argument("--save_dir", type=str, default="model_checkpoint.pth", help="File path to save the trained model checkpoint")
  parser.add_argument("--arch", type=str, default="inception_v3", help="Architecture of the neural network vgg13/Inception_v3")
  parser.add_argument("--gpu",type=bool, default=True)
   
  args = parser.parse_args()
  return args

def training(gpu,data_dir="flowers",save_dir="model_checkpoint.pth",arch="inception_v3",learning_rate=0.02,hidden_units=550,epochs=8):
    #Data Directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    #labels
    with open('cat_to_name.json', 'r') as f:
      cat_to_name = json.load(f)
    
    train_transforms =transforms.Compose([transforms.RandomRotation(30),
                        transforms.RandomResizedCrop(299),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])])
    validation_transforms = transforms.Compose([transforms.Resize(255),
                      transforms.CenterCrop(299),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                      transforms.CenterCrop(299),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])]) 
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms) 
    #Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validtnloader=torch.utils.data.DataLoader(validation_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
  #   for inputs, labels in trainloader:
  #     print("Batch Shape:", inputs.shape)
  #     print("Labels:", labels)
  #     break
    #shift to gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() and gpu==True else "cpu")
    print(device)
    
    #Create Model
    
    if(arch=='VGG'):
      model = models.vgg16(pretrained=True)

      for param in model.parameters():
        param.requires_grad = False
    
      model.classifier = nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(25088, 4096)),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(4096, 500)),
                          ('relu2', nn.ReLU()),
                        ('dropout',nn.Dropout(0.5)),
                          ('fc3', nn.Linear(500, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                        ]))
      optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
    
    elif (arch=="inception"):
      model=models.inception_v3(preTrained=True)
      
      for param in model.parameters():
        param.requires_grad = False
    
      model.fc = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(model.fc.in_features, hidden_units)),
                ('dropout',nn.Dropout(0.5)),
                ('fc2', nn.Linear(hidden_units, 102)),
                ('output', nn.LogSoftmax(dim=1))
                ]))
      optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    #print(model)

    criterion = nn.NLLLoss()
    
    model.to(device)
    #Training
    train_losses = []
    test_losses = []
    epochs_t = epochs
    steps = 0
    running_loss = 0
    print_every = 5

    for epoch in range(epochs_t):
      for inputs, labels in trainloader:
        # print(type(inputs))
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        logps = model.forward(inputs)
        print(type(logps))
        if(type(logps)==tuple):
          loss = criterion( logps[0], labels)
        else:
          loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if steps % print_every == 0:
          test_loss = 0
          accuracy = 0
          model.eval()
          with torch.no_grad():
            for inputs, labels in validtnloader:
              inputs, labels = inputs.to(device), labels.to(device)
              logps = model.forward(inputs)
              if(type(logps)==tuple):
                loss = criterion( logps[0], labels)
              else:
                loss = criterion( logps, labels)
              
              batch_loss = criterion(logps, labels)

              test_loss += batch_loss.item()

              # Calculate accuracy
              ps = torch.exp(logps)
              top_p, top_class = ps.topk(1, dim=1)
              equals = top_class == labels.view(*top_class.shape)
              accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            avg_train_loss = running_loss / print_every
            avg_test_loss = test_loss / len(validtnloader)
            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)

            print(f"Epoch {epoch+1}/{epochs_t}.. "
            f"Train loss: {avg_train_loss:.3f}.. "
            f"Validation loss: {avg_test_loss:.3f}.. "
            f"Validation accuracy: {accuracy/len(validtnloader):.3f}")
          running_loss = 0
          model.train()

    #Save the checkpoint 

    checkpoint = {'input_size': 3,
        'output_size': 102,
        'class_to_idx': train_data.class_to_idx,
          'epochs': epochs_t,
          'optimizer_state_dict': optimizer.state_dict(),
        'state_dict': model.state_dict()
        }
    torch.save(checkpoint, save_dir)

if __name__ == "__main__":
    
    args = parse_args()
    # Access the arguments using args.data_dir, args.learning_rate, etc.
    print("Data Directory:", args.data_dir)
    print("Learning Rate:", args.learning_rate)
    print("Hidden Units:", args.hidden_units)
    print("Epochs:", args.epochs)
    print("Save Directory:", args.save_dir)
    print("Architecture:", args.arch)
    if(args.gpu=="gpu"):
      gpu=True
    else:
      gpu=False
    # Call your training function with the provided arguments
    training(gpu,args.data_dir, args.save_dir,args.arch, args.learning_rate, args.hidden_units, args.epochs)