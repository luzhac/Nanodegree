#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import smdebug.pytorch as smd
import torchvision.datasets as datasets
#from torchvision.models import efficientnet_b0

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True 

import argparse

import os

# Access training and testing directories dynamically

#TODO: Import dependencies for Debugging andd Profiling

def test(model, test_loader,criterion,device,hook):
    '''
     Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    hook.set_mode(smd.modes.EVAL)


    print("Testing Model on Whole Testing Dataset")

    running_loss=0
    running_corrects=0
    
    for inputs, labels in test_loader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    test_loss = running_loss / len(test_loader.dataset)
    test_acc = running_corrects/ len(test_loader.dataset)
    print(f"Testing Loss: {test_loss:.4f}")
    print(f"Testing Accuracy: {test_acc:.4f}")



def train(model, train_loader, criterion, optimizer, device,epochs,hook):
    '''
     Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    hook.set_mode(smd.modes.TRAIN)


 
    best_loss=1e6

    loss_counter=0

    for epoch in range(epochs):
        for phase in ['train']:
            print(f"Epoch {epoch}, Phase {phase}")
            
            model.train()
           
            running_loss = 0.0
            running_corrects = 0
            running_samples=0

            for inputs, labels in train_loader:
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                   
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                running_samples+=len(inputs)
                if running_samples % 2000  == 0:
                    accuracy = running_corrects/running_samples
                    print("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                            running_samples,
                            len(train_loader)* inputs.size(0),
                            100.0 * (running_samples / (len(train_loader)* inputs.size(0))),
                            loss.item(),
                            running_corrects,
                            running_samples,
                            100.0*accuracy,
                        )
                    )                

            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects / running_samples

                   
    return model
    
def net():
    '''
    Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained=True)

    #model = efficientnet_b0(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 133))
    return model


def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    pass

def main(args):
    
    
    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}

    epochs =  args.epochs
    print(train_kwargs)
    print(epochs,args.lr)
   

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")
    
    '''
     Initialize a model by calling the net function
    '''
    model=net()
    model=model.to(device)
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    '''
     Create your loss and optimizer
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(),args.lr)
    
    
    '''
    Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    # Transformation for images
    transform = transforms.Compose([    transforms.Resize((224, 224)),    transforms.ToTensor()])

    
    # Path to your local dataset
    train_dir = os.getenv("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    test_dir = os.getenv("SM_CHANNEL_TEST", "/opt/ml/input/data/test")
    #train_dir =  "/home/sagemaker-user/CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter/data/train"
    #test_dir = "/home/sagemaker-user/CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter/data/test"

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory {train_dir} does not exist!")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Testing directory {test_dir} does not exist!")


    # Use ImageFolder to load the dataset
    trainset = datasets.ImageFolder(root=train_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, **train_kwargs,shuffle=True)
  
    # Use ImageFolder to load the dataset
    testset = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset,**test_kwargs)

     
    train(model, train_loader, criterion, optimizer, device,epochs,hook)
    
    '''
     Test the model to see its accuracy
    '''
    test(model, test_loader, criterion, device,hook)
    
    '''
     Save the trained model
    '''
    #torch.save(model.state_dict(),"/opt/ml/model/model.pth")
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, "/opt/ml/model/model.pth")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    Specify any training args that you might need
    '''
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.006, metavar="LR", help="learning rate (default: 1.0)"
    )
    
    args=parser.parse_args()
    
    main(args)
