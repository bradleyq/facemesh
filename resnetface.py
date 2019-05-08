import torch
import torchvision
import torchvision.transforms as transforms
import resnet
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import resnet

### Dataset class for loading face image mesh pairs ###
class FaceDataset(torch.utils.data.Dataset):

    def __init__(self, transform, train=True):
        self.image_prefix = "face_renders/face"
        self.image_suffix = ".jpg"
        self.vertex_prefix = "processed_faces/face"
        self.vertex_suffix = ".txt"
        self.count = 5000
        self.trainn = 4500
        
        self.train = train
        self.transform = transform
        
        shape = np.loadtxt(self.vertex_prefix + str(1) + self.vertex_suffix).shape
        tmp = np.zeros((self.count, shape[0], shape[1]))
        for i in range(self.count):
            tmp[i] = np.loadtxt(self.vertex_prefix + str(i + 1) + self.vertex_suffix)
            
        self.mean = np.mean(tmp, axis=0)
        self.outputdim = shape[0] * shape[1]
        self.labels = [torch.from_numpy((lab - self.mean).reshape(self.outputdim)).float() for lab in tmp]
            
        # simple version for working with CWD
        

    def __len__(self):
        if self.train:
            return self.trainn
        else:
            return self.count - self.trainn

    def __getitem__(self, idx):
        if not train:
            idx += self.trainn
        y = self.labels[idx]
        x = plt.imread(self.image_prefix + str(idx + 1) + self.image_suffix)
        
        sample = (x,y)
        sample = (self.transform(sample[0]), sample[1])

        return sample


def run(batchsize = 25, rate = 0.01, epochs = 50, lr_decay = 1.0, lr_stride = 1):

    ### create dataloaders ###
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = FaceDataset(transform, train=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=0)

    testset = FaceDataset(transform, train=False)
    testloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("torch.cuda.is_available()   =", torch.cuda.is_available())
    print("torch.cuda.device_count()   =", torch.cuda.device_count())
    print("torch.cuda.device('cuda')   =", torch.cuda.device(0))
    print("torch.cuda.current_device() =", torch.cuda.current_device())

    ### loading data to gpu if possible ###
    def to_device(data, device):
        if isinstance(data, (list, tuple)):
            return [to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)

    class DeviceDataLoader():
        def __init__(self, dl, device):
            self.dl = dl
            self.device = device
        
        def __iter__(self):
            for b in self.dl:
                yield to_device(b, self.device)
        
        def __len__(self):
            return len(self.dl)
        
    trainloader = DeviceDataLoader(trainloader, device)
    testloader = DeviceDataLoader(testloader, device)

    ### define model ###
    model = resnet.resnet18(output_size=trainset.outputdim)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=rate)

    criterion = nn.MSELoss()

    def adjust_learning_rate(optimizer, epoch, decay, stride):
        lr = rate * (decay ** (epoch // stride))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train(model, optimizer, criterion, epochs, trainloader, testloader):
        model.train()
        samples = 1
        losses = []
        test_losses = []
        k = len(trainloader)// samples
        
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % k == k - 1:

                    losses.append(running_loss / k)
                    
                    testloss = 0
                    total = 0
                    iterations = 0
                    with torch.no_grad():
                        for data in testloader:
                            images, labels = data
                            outputs = model(images)
                            testloss += criterion(outputs, labels)
                            total += labels.size(0)
                            iterations += 1
                            if total > 200:
                                break
                    test_losses.append(testloss / iterations)
                    
                    print('[%d, %5d] loss: %.3f test_loss: %.3f' %(epoch + 1, i + 1,losses[-1],test_losses[-1]))

                    running_loss = 0.0
                    
            adjust_learning_rate(optimizer, epoch+1, lr_decay, lr_stride)

        print('Finished Training')
        plt.plot(np.arange(0, len(losses)/samples, 1.0/samples), losses)
        plt.title("loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.show()
        
        plt.plot(np.arange(0, len(test_losses)/samples, 1.0/samples), test_losses)
        plt.title("test_loss")
        plt.xlabel("epoch")
        plt.ylabel("test_losses")
        plt.show()

    ### Begin Training ###
    train(model, optimizer, criterion, epochs, trainloader, testloader)
    return model

### Save the model weights ###
def save_model(model, batchsize, rate, epochs, lr_decay, lr_stride):
    torch.save(model.state_dict(), "res18b" + str(batchsize) + "r" + str(rate) + "e" + str(epochs) + ".statedict")

