from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import data
import gc
import os

gpu = True

class LSTM_Network(nn.Module):
    def __init__(self, input_features=19, hidden_size=128, num_layers=1, num_classes=10):
        super(LSTM_Network, self).__init__()
        self.num_classes = num_classes
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        #configure the layers
        self.lstm = nn.LSTM(input_size=input_features, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.activation = nn.ReLU()
        self.dense = nn.Linear(in_features=hidden_size, out_features=
            num_classes)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, x):
        out, _status = self.lstm(x)
        out = out[:, -1, :]
        out = self.dense(out)
        out = self.softmax(out)
        return out

class FeedForward_Network(nn.Module):
    def __init__(self, input_features=19, hidden_size = 128, num_layers=5, num_classes=10):
        super(FeedForward_Network, self).__init__()
        self.num_classes = num_classes
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        #configure the layers
        self.dense_layers = nn.ModuleList([])
        self.activation = nn.ReLU()
        self.first_layer = nn.Linear(in_features=input_features, out_features=hidden_size)
        for i in range(num_layers):
            self.dense_layers.append(nn.Linear(in_features=hidden_size, out_features=hidden_size))
        self.last_layer = nn.Linear(in_features = hidden_size, out_features=num_classes)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, x):
        out = self.first_layer(x)
        out = self.activation(out)
        for i in range(self.num_layers):
            out = self.dense_layers[i](out)
            out = self.activation(out)
        out = self.last_layer(out)
        out = self.softmax(out)
        return out

class Dataset(data.Dataset):
    def __init__(self, root, training, gpu):
        self.root = root
        #self.data_root = root + '/data/'
        #self.label_root = root + '/labels/'
        #self.data_names = os.listdir(self.data_root)
        self.device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
        self.data = torch.Tensor(np.load(root + '/data.npy')).to(self.device)
        self.label = torch.Tensor(np.load(root + '/labels.npy')).to(self.device)
        length = len(self.label)
        if training:
            self.data = self.data[:int(0.8 * length), :]
            self.label = self.label[:int(0.8 * length)]
        else:
            self.data = self.data[int(0.8 * length):, :]
            self.label = self.label[int(0.8 * length):]
        self.training = training
        self.index = 0

    def __getitem__(self, index):
        #data_name = self.data_names[index]
        #data_dir = self.data_root + data_name
        #label_dir = '%s%s.npy' % (self.label_root, data_name[:-4])
        data = self.data[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.label)

class AverageMeter(object):
    # Computes and stores the average and current value
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_dataloader(root, batch_size):
    train_set = Dataset(root, True, gpu)# + '/train', True)
    valid_set = Dataset(root, False, gpu)# + '/validation', False)

    train_loader = data.DataLoader(train_set, batch_size=batch_size, num_workers=0, drop_last=False)
    valid_loader = data.DataLoader(valid_set, batch_size=batch_size, num_workers=0, drop_last=False)

    return train_loader, valid_loader

def accuracy(predictions, labels):
    #correct = predictions.eq(labels.cpu()).sum().item()
    correct = predictions.eq(labels).sum().item()
    acc = correct/np.prod(labels.shape)
    return acc

def train_epoch(model, criterion, optimizer, trainloader, scaler, epoch, use_amp=True, gpu=True):
    losses = AverageMeter('Loss', ':.4e')
    accs = AverageMeter('Accuracy', ':6.4f')
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    for t, (inputs, labels) in enumerate(tqdm(trainloader, desc='Epoch ' + str(epoch))):
        inputs, labels = inputs.to(device), labels.to(device).long()

        if gpu and torch.cuda.is_available() and use_amp:
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(inputs)
                predictions = outputs.argmax(1)#.cpu().argmax(1)
                loss = criterion(outputs, labels)
        else:
            outputs = model(inputs)
            predictions = outputs.argmax(1)
            try:
                loss = criterion(outputs, labels)
            except:
                print(inputs)
                print(outputs)
                print(labels)
                exit()

        accs.update(accuracy(predictions, labels), inputs.shape[0])
        losses.update(loss.item(), inputs.size(0))

        if torch.cuda.is_available() and gpu and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        del inputs
        del labels
        del outputs
        gc.collect()
        if torch.cuda.is_available() and gpu:
            torch.cuda.empty_cache()

    print('--- training result ---')
    print('loss: %.5f, accuracy: %.5f' % (losses.avg, accs.avg))
    return losses.avg, accs.avg


def evaluate(model, criterion, validloader, test_flag=False, gpu=True, save_dir=None, stats_file=None):
    torch.cuda.empty_cache()
    losses = AverageMeter('Loss', ':.5f')
    accs = AverageMeter('Accuracy', ':6.4f')
    # Define confusion matrix via built in PyTorch functionality
    #conf_meter = ConfusionMeter(4)
    with torch.no_grad():
        model.eval()
        for t, (inputs, labels) in enumerate(tqdm(validloader)):
            if torch.cuda.is_available() and gpu:
                inputs, labels = inputs.cuda(), labels.cuda().long()
            else:
                inputs, labels = inputs, labels.long()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(1)

            losses.update(loss.item(), inputs.size(0))
            accs.update(accuracy(predictions, labels), inputs.shape[0])
            #conf_meter.add(outputs.permute(0, 2, 3, 4, 1).contiguous().view(-1, 2), labels.view(-1))
            del inputs
            del labels
            del outputs
            gc.collect()
            if torch.cuda.is_available() and gpu:
                torch.cuda.empty_cache()
        print('--- evaluation result ---')
        print('loss: %.5f, accuracy: %.5f' % (losses.avg, accs.avg))
        print()
    return losses.avg, accs.avg

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() and gpu else 'cpu'

    criterion = nn.CrossEntropyLoss().to(device)
    model = FeedForward_Network().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)
    train_loader, valid_loader = get_dataloader(root='cleaned_data/cole', batch_size=128)
    scheduler = ReduceLROnPlateau(optimizer, patience=20)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    for epoch in range(1, 10 + 1):
            loss_train, acc_train = train_epoch(model, criterion, optimizer, train_loader, scaler, epoch, use_amp=False, gpu=gpu)
            loss_val = evaluate(model, criterion, valid_loader, gpu=gpu)
            if not os.path.isdir('saved/%s' % ('feedforward/')):
                os.mkdir('saved/%s' % ('feedforward'))
            torch.save({
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'model_state_dict': model.state_dict(),
                'loss_val': loss_val,
                'loss_train': loss_train,
            }, 'saved/%s/%s' % ('feedforward', str(epoch) + '.pth' ))

            scheduler.step(loss_train)
