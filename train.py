import torch
from dataset import get_cifar10
from model import ResNet20

def train_loop(model, loss_fn, optimizer,  train_loader, val_loader, device, n_epoch=500):
    best_val_acc = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    model.to(device)

    for epoch in range(n_epoch):

        model.train()

        epoch_train_loss = 0
        epoch_train_acc = 0
        for batch_idx, (data, target) in enumerate(train_loader):

            data = data.to(device)
            target = target.to(device)

            output = model(data)

            loss = loss_fn(output, target)
            predictions = output.data.max(1)[1]
            epoch_train_acc += predictions.eq(target).sum().item()
            epoch_train_loss += loss.item()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        train_losses.append(epoch_train_loss / len(train_loader.dataset))
        train_accuracies.append(epoch_train_acc / len(train_loader.dataset))



        model.eval()

        epoch_val_loss = 0
        epoch_val_acc = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):

                data = data.to(device)
                target = target.to(device)

                output = model(data)

                loss = loss_fn(output, target)
                predictions = output.data.max(1)[1]
                epoch_val_acc += predictions.eq(target).sum().item()
                epoch_val_loss += loss.item()

            val_losses.append(epoch_val_loss / len(val_loader.dataset))
            val_accuracies.append(epoch_val_acc / len(val_loader.dataset))


        print(f'[EPOCH]: {epoch}, [TRAIN LOSS]: {train_losses[-1]}, [TRAIN ACCURACY]: {train_accuracies[-1]}')
        print(f'[EPOCH]: {epoch}, [VAL LOSS]: {val_losses[-1]}, [VAL ACCURACY]: {val_accuracies[-1]}')

        if val_accuracies[-1] > best_val_acc:
            best_val_acc = val_accuracies[-1]
            print('Saving best model')
            torch.save(model.state_dict(), 'best_model2')
        print()

    return train_losses, train_accuracies, val_losses, val_accuracies



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, val_loader, _ = get_cifar10()

model = ResNet20(3, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

train_losses, train_accuracies, val_losses, val_accuracies = train_loop(model, loss_fn, optimizer, train_loader, val_loader, device)
