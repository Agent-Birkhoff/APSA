import torch
import torch.utils.data.dataloader as dataloader
import torchvision
from torch import cuda, nn, optim
from torch.autograd.grad_mode import no_grad
from torchvision import datasets, transforms
from tqdm import tqdm

from apsaRES import ApsaNet
from apsaSWIN import APSA_SWIN_B


def main(batch_size=16, num_worker=4, epoch=100, lr=0.0001):
    trainset = datasets.CIFAR100(
        "./cifar100",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        ),
    )
    testset = torchvision.datasets.CIFAR100(
        "./cifar100",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ],
        ),
    )

    trainloader = dataloader.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_worker,
        pin_memory=True,
        drop_last=True,
    )
    testloader = dataloader.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_worker,
        pin_memory=True,
        drop_last=True,
    )

    model = APSA_SWIN_B(num_classes=100).cuda()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.9)

    bar = tqdm(range(1, epoch + 1))
    for i in bar:
        predicted = []
        gt = []
        for x, y in tqdm(trainloader):
            x = x.cuda()
            y = y.cuda()

            optimizer.zero_grad()
            y_p = model(x)

            predicted_i = y_p.argmax(dim=-1)
            correct = predicted_i == y
            acc = correct.float().mean() * 100
            gt += y.tolist()
            predicted += predicted_i.tolist()

            loss = loss_fn(y_p, y)
            bar.set_description(
                "(train) epoch{}, train loss: {:.3f}, acc: {:f}%".format(
                    i, loss.item(), acc
                )
            )
            loss.backward()
            optimizer.step()
        correct = [predicted[n] == gt[n] for n in range(len(gt))]
        acc = sum(correct) / len(gt) * 100
        print("(train) epoch{}, acc: {:.2f}%".format(i, acc))

        predicted = []
        gt = []
        with no_grad():
            for x, y in tqdm(testloader):
                x = x.cuda()
                y = y.cuda()

                y_p = model(x)

                predicted_i = y_p.argmax(dim=-1)
                correct = predicted_i == y
                acc = correct.float().mean() * 100
                gt += y.tolist()
                predicted += predicted_i.tolist()

                loss = loss_fn(y_p, y)
                bar.set_description(
                    "(test) epoch{}, loss: {:.3f}, acc: {:.2f}%".format(
                        i, loss.item(), acc
                    )
                )
        correct = [predicted[n] == gt[n] for n in range(len(gt))]
        acc = sum(correct) / len(gt) * 100
        print("(test) epoch{}, acc: {:.2f}%".format(i, acc))


cuda.set_device(1)
main()
