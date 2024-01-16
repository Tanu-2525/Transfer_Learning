import torch
print(torch.__version__)
# print(torch.cuda.is_available())
from collections import OrderedDict
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn 
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim
from torchvision import transforms, models
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

def parse_args():
  parser = argparse.ArgumentParser(description="Predict a flower")
  parser.add_argument("--checkpoint_pth", type=str,default="model_checkpoint.pth", help="Model checkpoint")
  parser.add_argument("--flower_path", type=str, default="flowers/test/100/image_07896,jpg", help="Path to flower ")
  parser.add_argument("--arch", type=str, default="inception_v3", help="Architecture of the neural network (e.g., vgg13)")
  parser.add_argument("--num_probab", type=int, default="5", help="Number of Probabilities to show")
  parser.add_argument("--json_path", type=str, default="cat_to_name.json", help="Path to json file")
  parser.add_argument("--gpu",type=bool, default=True)
   
  args = parser.parse_args()
  return args


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
        
    '''
    # Load the image using PIL
    img = Image.open(image)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    # Resize the image while keeping the aspect ratio
    img.thumbnail((256, 256))

    # Crop the center 224x224 portion of the image
    img = transforms.CenterCrop(224)(img)

    # Convert PIL image to Numpy array
    np_image = np.array(img)

    # Convert integer values to floats in the range [0, 1]
    np_image = np_image / 255.0

    # Normalize the image
    np_image = np.array(img) / 255.0  # Convert to float in the range [0, 1]

    # Normalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    # Transpose dimensions to match PyTorch expectations
    np_image = np_image.transpose(2, 0, 1)
    return np_image
   
def load_checkpoint(checkpoint_path,arch):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    # Define input and output sizes based on the checkpoint
    input_size = checkpoint['input_size']
    output_size = checkpoint['output_size']

    if(arch=='VGG'):
      model_trained = models.vgg16(pretrained=True)

      for param in model.parameters():
        param.requires_grad = False
    
      model_trained.classifier = nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(25088, 4096)),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(4096, 500)),
                          ('relu2', nn.ReLU()),
                        ('dropout',nn.Dropout(0.5)),
                          ('fc3', nn.Linear(500, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                        ]))
     
    elif (arch=="inception"):
      model_trained=models.inception_v3(preTrained=True)
      
      for param in model.parameters():
        param.requires_grad = False
    
      model_trained.fc = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(model_trained.fc.in_features, 550)),
                ('dropout',nn.Dropout(0.5)),
                ('fc2', nn.Linear(550, 102)),
                ('output', nn.LogSoftmax(dim=1))
                ]))


    # Load the model's state_dict
    model_trained.load_state_dict(checkpoint['state_dict'])
    # Attach additional information to the model
    model_trained.class_to_idx = checkpoint['class_to_idx']
    # Access other information
    epochs = checkpoint['epochs']
    optimizer = optim.Adam(model_trained.classifier.parameters(), lr=0.003)  
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model_trained, optimizer, epochs

def predict_flower(image_path, model,class_to_idx, topk,gpu=True):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() and gpu==True else "cpu")
    print(device)
    model.to(device)
    processed_image = process_image(image_path)
    input_tensor = torch.from_numpy(processed_image)
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(torch.float)
    # Make sure to send the input tensor to the same device as your model
    input_tensor = input_tensor.to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)        
    probabilities, indices = torch.topk(output, topk)
    # Convert indices to class labels using the inverted class_to_idx dictionary
    idx_to_class = {val: key for key, val in class_to_idx.items()}
    top_classes = [idx_to_class[idx.item()] for idx in indices[0]]
    # Convert probabilities to a list
    top_probabilities = torch.nn.functional.softmax(probabilities[0], dim=0).tolist()
    return top_probabilities, top_classes

def predict_main(checkpoint_path,test_image_pth,arch,top_k,json_path,gpu):
    loaded_model, loaded_optimizer, loaded_epochs = load_checkpoint(checkpoint_path,arch)
    class_to_idx = loaded_model.class_to_idx
    processed_image = process_image(test_image_pth)
    top_probabilities, top_classes = predict_flower(test_image_pth,loaded_model,class_to_idx, top_k, gpu)
    print("Top classes and probabilities:",top_classes,":",top_probabilities)


if __name__ == "__main__":
    args = parse_args()
    # Call predict function with the provided arguments
    if(args.gpu=="gpu"):
        gpu=True
    else:
       gpu=False
    predict_main(args.checkpoint_pth, args.flower_path, args.arch, args.num_probab,args.json_path, gpu)
    