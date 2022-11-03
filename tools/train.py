import logging
logging.basicConfig(level=logging.INFO)
from tqdm import tqdm
import torch
torch.manual_seed(0)
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Resize, Normalize, ToTensor

import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(__file__)))
from datasets.caltech256 import Caltech256
from models.mobilenet_v1 import MobileNetV1

def train_1epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):
    batch_acc = []
    batch_loss = []
    model.train()
    with torch.enable_grad():
        for data, target in dataloader:
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            output = model(data)    # B x num_classes
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            probs = torch.softmax(output, dim=1)
            preds = probs.argmax(dim=1)
            acc = torch.sum(preds == target)
            batch_acc.append(acc)
            batch_loss.append(loss)

    avg_acc = sum(batch_acc) / len(batch_acc)
    avg_loss = sum(batch_loss) / len(batch_loss)

    return avg_acc, avg_loss

def validate_1epoch(
    model,
    dataloader,
    criterion,
    device
):
    batch_acc = []
    batch_loss = []
    model.eval()
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            probs = torch.softmax(output, dim=1)
            preds = probs.argmax(dim=1)
            acc = torch.sum(preds == target)
            batch_acc.append(acc)
            batch_loss.append(loss)

    avg_acc = sum(batch_acc) / len(batch_acc)
    avg_loss = sum(batch_loss) / len(batch_loss)

    return avg_acc, avg_loss  


if __name__ == '__main__':
    # check gpu availability.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # resize to 224x224.
    trans_X = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # load dataset.
    dataset = Caltech256(trans_X)
    dataloader = DataLoader(dataset)

    # split train and test.
    n = len(dataset)
    n_train = int(n * 0.8)
    n_test = n - n_train
    train_dataset, test_dataset = random_split(dataset, [n_train, n_test])

    # create dataloader.
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # create model.
    model = MobileNetV1(257)
    model.to(device)

    # create loss function
    criterion = torch.nn.CrossEntropyLoss()

    # create optimizer
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)

    # loop run epoch
    best_loss = float('inf')
    best_acc = -float('inf')
    writer = SummaryWriter('logs')
    for i in tqdm(range(100)):
        train_acc, train_loss = train_1epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_loss = validate_1epoch(model, train_loader, criterion, device)
        logging.info(f'epoch:{i}, train loss:{train_loss}, train acc:{train_acc}, val loss{val_loss}, val acc:{val_acc}')
        writer.add_scalar('Acc/train', train_acc, i+1)
        writer.add_scalar('Acc/val', val_acc, i+1)
        writer.add_scalar('Loss/train', train_loss, i+1)
        writer.add_scalar('Loss/val', train_loss, i+1)
        if best_acc < val_acc:
            torch.save(model.state_dict(), 'best_acc.pth')
            best_acc = val_acc
        if best_loss > val_loss:
            torch.save(model.state_dict(), 'best_loss.pth')
            best_loss = val_loss

    writer.close()