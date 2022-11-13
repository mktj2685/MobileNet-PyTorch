# import my packages and modules.
import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(__file__)))
from datasets.caltech256 import Caltech256
from models.mobilenet_v1 import MobileNetV1
import argparse
import logging
logging.basicConfig(level=logging.INFO)
from tqdm import tqdm
import torch
torch.manual_seed(0)
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Resize, Normalize, ToTensor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=100, type=int, help='number of epochs.')
    parser.add_argument('--batch_size', default=1, type=int, help='how many samples per batch to load.')

    return parser.parse_args()

def train_1epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):
    epoch_acc = 0.
    epoch_loss = 0.
    model.train()
    for data, target in dataloader:
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        probs = torch.softmax(output, dim=1)
        preds = probs.argmax(dim=1)
        epoch_acc += torch.sum(preds == target).item()
        epoch_loss += loss.item()
    epoch_acc = 100 * epoch_acc / len(dataloader.dataset)
    epoch_loss = epoch_loss / len(dataloader)

    return epoch_acc, epoch_loss

def validate_1epoch(
    model,
    dataloader,
    criterion,
    device
):
    epoch_acc = 0.
    epoch_loss = 0.
    model.eval()
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            preds = probs.argmax(dim=1)
            epoch_acc += torch.sum(preds == target).item()
            epoch_loss += criterion(output, target).item()
    epoch_acc = 100 * epoch_acc / len(dataloader.dataset)
    epoch_loss = epoch_loss / len(dataloader)

    return epoch_acc, epoch_loss  


if __name__ == '__main__':
    # parse arguments.
    args = parse_args()

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

    # split train and test.
    n = len(dataset)
    n_train = int(n * 0.8)
    n_test = n - n_train
    train_dataset, test_dataset = random_split(dataset, [n_train, n_test])

    # create dataloader.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

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
        val_acc, val_loss = validate_1epoch(model, test_loader, criterion, device)
        logging.info(
            f'epoch:{i},                    \
            train loss:{train_loss:.2f},    \
            train acc:{train_acc:.2f},      \
            val loss:{val_loss:.2f},        \
            val acc:{val_acc:.2f}'
        )
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