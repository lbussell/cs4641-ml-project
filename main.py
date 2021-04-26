from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import data
import gc
import os
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial

gpu = True

class LSTM_Network(nn.Module):
    def __init__(self, input_features=14, hidden_size=128, num_layers=1, num_hidden=1, num_classes=2):
        super(LSTM_Network, self).__init__()
        self.num_classes = num_classes
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        #configure the layers
        self.lstm = nn.LSTM(input_size=input_features, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.activation = nn.LeakyReLU()
        self.hidden_layers = nn.ModuleList([])
        self.hidden_dropout = nn.ModuleList([])
        for i in range(num_hidden):
            self.hidden_layers.append(nn.Linear(in_features=hidden_size, out_features=
            hidden_size))
            self.hidden_dropout.append(nn.Dropout(p=0.2))
        self.denseout = nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, x, hidden):
        out, hidden_out = self.lstm(x, hidden)
        #out is of the form [batch, pitch, pitch probabilities]
        out = out[:, -1, :]
        for i in range(len(self.hidden_layers)):
            out = self.hidden_layers[i](out)
            out = self.activation(out)
            out = self.hidden_dropout[i](out)
        out = self.denseout(out)
        out = self.softmax(out)
        return out
    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, batch_size, self.hidden_size).cuda(),
                    weight.new_zeros(self.num_layers, batch_size, self.hidden_size).cuda())

class FeedForward_Network(nn.Module):
    def __init__(self, input_features=13, hidden_size = 128, num_layers=5, num_classes=2):
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
        self.last_layer = nn.Linear(in_features=hidden_size, out_features=num_classes)
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
    def __init__(self, root, training, gpu, test=False):
        self.root = root
        self.data_root = root + '/data/'
        self.label_root = root + '/labels/'
        self.data_names = os.listdir(self.data_root)
        self.device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
        self.training = training
        self.index = 0
        self.data = []
        self.labels = []
        for inning_name in self.data_names:
            inning = torch.Tensor(np.load(self.data_root + inning_name)).to(self.device)
            label = torch.Tensor(np.load(self.label_root + inning_name)).to(self.device)
            for i in range(len(label) - 5):
                sequence = inning[i: i + 5]
                self.data.append(sequence)
                self.labels.append(label[i + 5])
        self.data = torch.stack(tuple(self.data)).to(self.device)
        self.labels = torch.Tensor(self.labels).to(self.device)
        length = len(self.labels)
        if training:
            self.data = self.data[:int(0.1 * length), :]
            self.labels = self.labels[:int(0.1 * length)]
        elif test:
            self.data = self.data[int(0.9 * length):, :]
            self.labels = self.labels[int(0.9 * length):]
            print(self.labels.sum()/self.labels.shape[0])
        else:
            self.data = self.data[int(0.7 * length):int(0.9 * length), :]
            self.labels = self.labels[int(0.7 * length):int(0.9 * length)]
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    def __len__(self):
        return len(self.labels)

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

    train_loader = data.DataLoader(train_set, batch_size=batch_size, num_workers=0, drop_last=True, shuffle=True)
    valid_loader = data.DataLoader(valid_set, batch_size=batch_size, num_workers=0, drop_last=True, shuffle=True)

    return train_loader, valid_loader

def accuracy(predictions, labels):
    #correct = predictions.eq(labels.cpu()).sum().item()
    correct = predictions.eq(labels).sum().item()
    acc = correct/np.prod(labels.shape)
    return acc

def train_epoch(model, criterion, optimizer, trainloader, scaler, epoch, use_amp=True, gpu=True, batch_size=64):
    losses = AverageMeter('Loss', ':.4e')
    accs = AverageMeter('Accuracy', ':6.4f')
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    for t, (inputs, labels) in enumerate(tqdm(trainloader, desc='Epoch ' + str(epoch))):
        hidden = model.init_hidden(batch_size)
        inputs, labels = inputs.to(device), labels.to(device).long()

        if gpu and torch.cuda.is_available() and use_amp:
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(inputs, hidden)
                predictions = outputs.argmax(1)#.cpu().argmax(1)
                loss = criterion(outputs, labels)
        else:
            outputs = model(inputs, hidden)
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


def evaluate(model, criterion, validloader, test_flag=False, gpu=True, save_dir=None, stats_file=None, batch_size=64):
    torch.cuda.empty_cache()
    losses = AverageMeter('Loss', ':.5f')
    accs = AverageMeter('Accuracy', ':6.4f')
    # Define confusion matrix via built in PyTorch functionality
    #conf_meter = ConfusionMeter(4)
    with torch.no_grad():
        model.eval()
        for t, (inputs, labels) in enumerate(tqdm(validloader)):
            hidden = model.init_hidden(batch_size)
            if torch.cuda.is_available() and gpu:
                inputs, labels = inputs.cuda(), labels.cuda().long()
            else:
                inputs, labels = inputs, labels.long()
            outputs = model(inputs, hidden)
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

def train(config, checkpoint_dir=None, cwd=None, tuning=False):
    device = 'cuda' if torch.cuda.is_available() and gpu else 'cpu'
    num_hidden = config['num_hidden']
    num_layers = config['num_layers']
    model = LSTM_Network(num_hidden=num_hidden, num_layers=num_layers, hidden_size=512).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    opt = config['opt']
    lr = config['lr']
    weight_decay = config['weight_decay']
    beta_1 = config['beta_1']
    beta_2 = config['beta_2']
    momentum = config['momentum']
    max_iter = config['max_iter']
    epochs = config['epoch']
    patience = config['patience']
    batch_size = config['batch_size']
    if opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta_1, beta_2))
    elif opt == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta_1, beta_2))
    elif opt == 'lbfgs':
        optimizer = optim.LBFGS(model.parameters(), lr=lr, max_iter=max_iter)
    elif opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    if checkpoint_dir:
        checkpoint = os.path.join(os.path.join(cwd, checkpoint_dir), 'checkpoint')
        if os.path.isfile(checkpoint):
            model_state, optimizer_state = torch.load(checkpoint)
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    batch_size = batch_size
    train_loader, valid_loader = get_dataloader(root=os.path.join(cwd, 'cleaned_data/kershaw'), batch_size=batch_size)
    scheduler = ReduceLROnPlateau(optimizer, patience=patience)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    #checkpoint = torch.load('saved/lstm/20.pth')
    #model.load_state_dict(checkpoint['model_state_dict'])
    for epoch in range(1, epochs + 1):
        loss_train, acc_train = train_epoch(model, criterion, optimizer, train_loader, scaler, epoch, use_amp=False, gpu=gpu, batch_size=batch_size)
        loss_val, acc_val = evaluate(model, criterion, valid_loader, gpu=gpu, batch_size=batch_size)
        #if not os.path.isdir('saved/%s' % ('lstm/')):
        #    os.mkdir('saved/%s' % ('lstm'))
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'model_state_dict': model.state_dict(),
            'loss_val': loss_val,
            'loss_train': loss_train,
        }, os.path.join(cwd, 'saved/%s/%s' % ('lstm', str(epoch) + '.pth' )))
        if tuning:
            with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, 'checkpoint')
                torch.save((model.state_dict(), optimizer.state_dict()), path)
            tune.report(loss=loss_val, accuracy=acc_val)
        
        if (optimizer.param_groups[0]['lr'] / config['lr']) <= 1e-3:
                print('Learning Rate Reduced to 1e-3 of Original Value', 'Training Stopped', sep='\n')
                break
        scheduler.step(loss_train)

def optimize():
    name_dir = os.path.join('saved', 'hyper-lstm')
    hyperparam_config = {
        'name': 'lstm',
        'num_hidden': tune.sample_from(lambda _: np.random.randint(1,10)),
        'num_layers': tune.sample_from(lambda _: np.random.randint(1,5)),
        'opt': tune.choice(['adam', 'sgd', 'adamw', 'lbfgs']),
        'lr': tune.loguniform(1e-10, 1),
        'epoch': tune.sample_from(lambda _: np.random.randint(5,25)),
        'beta_1': tune.loguniform(1e-8, 1e-2),
        'beta_2': tune.loguniform(1e-8, 1e-2),
        'weight_decay': tune.loguniform(1e-8, 1e-2),
        'max_iter': tune.sample_from(lambda _: np.random.randint(10,100)),
        'momentum': tune.uniform(0.5, 0.9),
        'patience': tune.sample_from(lambda _: np.random.randint(5, 25)),
        'batch_size': 16
    }
    if not os.path.isdir(name_dir):
        os.mkdir(name_dir)
    scheduler = ASHAScheduler(
            metric='accuracy',
            mode='max',
            max_t=25,
            grace_period=1,
            reduction_factor=2
    )
    reporter = CLIReporter(metric_columns=["loss", "accuracy"])
    result = tune.run(
            partial(train, checkpoint_dir=name_dir, cwd=os.getcwd(), tuning=True),
            resources_per_trial = {"cpu": 1, "gpu": 0.5},
            config = hyperparam_config,
            num_samples = 200,
            scheduler = scheduler,
            progress_reporter = reporter
    )
    best_trial = result.get_best_trial("accuracy", "max", "last")
    
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
    print("Best Checkpoint Dir: " + str(best_trial.checkpoint.value))
    return best_trial.config

def test(config):
    device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
    batch_size = 1
    test_set = Dataset('cleaned_data/kershaw', False, gpu, test=True)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, num_workers=0, drop_last=True, shuffle=True)
    checkpoint = torch.load('saved/%s/%s.pth' % (config['name'], config['epoch']))
    model = LSTM_Network(num_hidden=config['num_hidden'], num_layers=config['num_layers'], hidden_size=512).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    criterion = nn.CrossEntropyLoss().to(device)
    loss_val, acc_val = evaluate(model, criterion, test_loader, gpu=gpu, batch_size=batch_size)

if __name__ == '__main__':
    name_dir = os.path.join('saved', 'hyper-lstm')
    #old_config={'name': 'lstm', 'num_hidden': 4, 'num_layers': 7, 'opt': 'lbfgs', 'lr': 4.647114878517564e-06, 'epoch': 7, 'beta_1': 6.870079202867473e-06, 'beta_2': 1.3324095555115266e-06, 'weight_decay': 8.86976993946711e-06, 'max_iter': 86, 'momentum': 0.7597484977901667, 'patience': 18, 'batch_size': 16}
    config = optimize()
    train(config=config, cwd=os.getcwd(), checkpoint_dir=name_dir)
    test(config)

