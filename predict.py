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

def imshow(image, ax=None, title=None, normalize=True):
  """Imshow for Tensor."""
  if ax is None:
      fig, ax = plt.subplots()
  image = image.numpy().transpose((1, 2, 0))

  if normalize:
      mean = np.array([0.485, 0.456, 0.406])
      std = np.array([0.229, 0.224, 0.225])
      image = std * image + mean
      image = np.clip(image, 0, 1)

  ax.imshow(image)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['left'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.tick_params(axis='both', length=0)
  ax.set_xticklabels('')
  ax.set_yticklabels('')

  return ax

def plot_results(image_path, top_probabilities, top_classes, json_path):
    # Load the input image
    img = Image.open(image_path)
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(figsize=(8, 12), nrows=2)
    # Display the input image using imshow function
    imshow(process_image(image_path), ax=ax1)
#     ax1.set_title('Orange Dahlia')
    with open(json_path, 'r') as f:
      class_to_name = json.load(f)
    # Map class indices to flower names using the class_to_name dictionary
    class_names = [class_to_name[str(class_idx)] for class_idx in top_classes]
    # Plot the bar graph for the top 5 classes with flower names
    ax2.barh(range(len(top_probabilities)), top_probabilities, color='blue')
    ax2.set_yticks(range(len(top_probabilities)))
    ax2.set_yticklabels(class_names)
    ax2.set_xlabel('Probability')
    ax2.set_title('Top 5 Flower Probabilities')

    plt.tight_layout()
    plt.show()


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
    # Load the pre-trained VGG16 model
    model_trained = models.__dict__[arch](pretrained=True)
    # Freeze the parameters in the features part
    for param in model_trained.features.parameters():
        param.requires_grad = False
    # Modify the classifier
    classifier_n = nn.Sequential(OrderedDict([
              ('fc1', nn.Linear(model_trained.fc.in_features, 550)),
              ('dropout',nn.Dropout(0.5)),
              ('fc2', nn.Linear(550, 102)),
              ('output', nn.LogSoftmax(dim=1))
    ]))

    model_trained.classifier = classifier_n
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
    plot_results(test_image_pth, top_probabilities, top_classes,json_path)

if __name__ == "__main__":
    args = parse_args()
    # Call predict function with the provided arguments
    predict_main(args.checkpoint_pth, args.flower_path, args.arch, args.num_probab,args.json_path, args.gpu)
    