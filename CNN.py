import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# this class creates a convolutional neural network with 3 convolutional layers, 2 fully connected layers,
# a max pooling layer, and a ReLU activation function that takes in images with 3 channels
# that is designed for binary image classification tasks
class CNN(nn.Module):
    # initialize the network with convolutional, pooling, and fully connected layers
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # adjust first FC layer to line up with the formula (W-F+2P)/S + 1
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)
    # define the forward propogation of the network
    def forward(self, x):
        # apply the ReLU function, along with max pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten the size of the resulting tensor
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# create a transformation that resizes images to 128x128 and converts them to PyTorch tensors
trans = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
# take in the training and testing data and apply the transformations
training = datasets.ImageFolder(root='/Users/alexmann/Documents/Datas/FlightData/png_train',
                                     transform=trans)
testing = datasets.ImageFolder(root='/Users/alexmann/Documents/Datas/FlightData/png_test',
                                    transform=trans)

# set up DataLoaders for the testing and training data, ensuring batch size is small (4)
train_loader = DataLoader(training, batch_size=4, shuffle=True)
test_loader = DataLoader(testing, batch_size=4, shuffle=False)

# initialize an instance of the CNN class
model = CNN()
# make sure that the CPU of the local device is being used if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# initialize the loss criterion and optimizer for the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# this function creates a neural network based on the CNN class, trains the network on the training set,
# for a given number of epochs (default 10), uses the model to make predictions on the test set,
# then generates a classification report and confusion matrix
def run_model(epochs=10):
    # train the model over each epoch
    for epoch in range(epochs):
        model.train()
        # set a value for the running loss
        running_loss = 0.0
        # iterate over each of the images, training the model on the proper labels
        for images, labels in train_loader:
            # use the local device to process the images and labels
            images = images.to(device)
            labels = labels.to(device)
            # generate the output of the model using the given images
            outputs = model(images)
            # calculate the loss
            loss = criterion(outputs, labels)
            # zero the gradient
            optimizer.zero_grad()
            # apply back-propagation
            loss.backward()
            optimizer.step()
            # add the loss to the running loss
            running_loss += loss.item()
        # empty the cache at the end of the epoch
        torch.cuda.empty_cache()
        # keep a running tab of the epoch # and the overall loss
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')

    # evaluate the model on the test data
    model.eval()
    # keep a running list of the labels and predictions
    total_labs, total_preds = list(), list()
    with torch.no_grad():
        for images, labels in test_loader:
            # use the local device to process the images and labels
            images = images.to(device)
            labels = labels.to(device)
            # generate the output of the model based on the test images
            outputs = model(images)
            # generate the prediction and add it to the list
            _, preds = torch.max(outputs, 1)
            total_labs.extend(labels.cpu().numpy())
            total_preds.extend(preds.cpu().numpy())

    # calculate the total accuracy of the model
    accuracy = np.sum(np.array(total_preds) == np.array(total_labs)) / len(total_labs)
    print(f'Accuracy: {accuracy:.4f}')
    # print the classification report
    print(classification_report(total_labs, total_preds, target_names=['good', 'bad']))
    # print the confusion matrix
    cm = confusion_matrix(total_labs, total_preds)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['bad', 'good'])
    display.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

run_model()


