'''Train ImageNet with PyTorch.'''
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import wandb
from utils import progress_bar

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        if args.method in ["lr", "hybrid", "cosine", "poly", "exp"]:
            last_lr = scheduler.get_last_lr()[0]
            wandb.log({'last_lr': last_lr})

    training_acc = 100.*correct/total
    wandb.log({'training_acc': training_acc,
               'training_loss': train_loss/(batch_idx+1)})

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
            
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    wandb.log({'accuracy': acc})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--batchsize', default=32, type=int, help='training batch size')
    parser.add_argument('--epochs', default=200, type=int, help="the number of epochs")
    parser.add_argument('--decay_epoch', default=40, type=int, help="the number of epochs to decay leraning rate")
    parser.add_argument('--power', default=0.9, type=float, help="polinomial or exponential power")
    parser.add_argument('--method', default="batch", type=str, help="constant, lr, batch, hybrid, poly, cosine, exp")
    
    args = parser.parse_args()
    wandb_project_name = "new-sigma-nice"
    wandb_exp_name = f"ImageNet,{args.method},b={args.batchsize},lr={args.lr}"
    wandb.init(config = args,
               project = wandb_project_name,
               name = wandb_exp_name,
               entity = "XXXXXX")
    wandb.init(settings=wandb.Settings(start_method='fork'))

    print('==> Preparing data..')
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])

    trainset = torchvision.datasets.ImageFolder(root='./data/train', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)

    testset = torchvision.datasets.ImageFolder(root='./data/val', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, shuffle=True, num_workers=2)

    device = 'cuda:0'
    net = torchvision.models.resnet34(pretrained=False)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 1000)
    net = net.to(device)
    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.0)

    if args.method == "lr":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_epoch, gamma=0.707106781186548)
    elif args.method == "hybrid":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_epoch, gamma=0.866025403784439)
        increase = 1.5
    elif args.method == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    elif args.method == "poly":
        scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=200, power=args.power)
    elif args.method == "exp":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.power)
    elif args.method == "batch":
        increase = 2
    print(optimizer)

    next_batch = args.batchsize
    for epoch in range(args.epochs):
        train(epoch)
        test(epoch)
        if args.method in ["lr", "hybrid", "cosine", "poly", "exp"]:
            scheduler.step()
        if args.method in ["batch", "hybrid"]:
            wandb.log({'batch': next_batch})
            if epoch % args.decay_epoch == 0 and epoch != 0:
                next_batch = int(next_batch * increase)
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=next_batch, shuffle=True, num_workers=2)
