import numpy as np
import torch
from torch import nn
from torch.cuda import amp
from tqdm.auto import tqdm

from metrics import compute_metrics

from model import PepCNN
from data_generator import dataload


def trainning(n_epochs, patience, batch_size, data_path):
    criterion = nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, test_loader, length_itemset = dataload(
        batch_size=batch_size, data_path=data_path)
    model = PepCNN(gap_constraint=5, item_size=length_itemset).cuda()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0003, weight_decay=1e-5)
    stale = 0
    best_acc = 0
    acc_sum, n = 0.0, 0
    for epoch in tqdm(range(n_epochs)):

        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        model.train()

        # These are used to record information in training.
        train_loss = []
        train_accs = []

        for batch in train_loader:
            imgs, labels = batch
            # ----------- mix-up -----------
            alpha = 2
            lam = np.random.beta(alpha, alpha)
            index = torch.randperm(imgs.size(0)).cuda()
            inputs = lam * imgs + (1 - lam) * imgs[index, :]
            targets_a, targets_b = labels.to(device), labels[index].to(device)
            outputs = model(inputs.to(device))
            loss = lam * criterion(outputs, targets_a) + \
                (1 - lam) * criterion(outputs, targets_b)
            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()

            # Compute the gradients for parameters.
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=10)

            # Update the parameters with computed gradients.
            optimizer.step()
            # Compute the accuracy for current batch.

            acc_sum += (outputs.argmax(dim=1) ==
                        labels.to(device)).float().sum().item()
            n += labels.shape[0]
            train_loss.append(loss.item())

        train_loss = sum(train_loss) / len(train_loss)

        # Print the information.
        print(
            f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {acc_sum/n:.5f}")

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()

        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []

        # Iterate the validation set by batches.
        acc_sum, n = 0.0, 0
        for batch in test_loader:
            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            # imgs = imgs.half()

            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs.to(device))

            # We can still compute the loss (but not the gradient).
            loss = criterion(logits, labels.to(device))

            # Compute the accuracy for current batch.

            # Record the loss and accuracy.
            valid_loss.append(loss.item())

            acc_sum += (logits.argmax(dim=1) ==
                        labels.to(device)).float().sum().item()
            n += labels.shape[0]
            # break
        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)

        # Print the information.
        print(
            f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {acc_sum / n:.5f}")

        # save models
        if acc_sum / n > best_acc:
            print(f"Best model found at epoch {epoch}, saving model")
            # only save best to prevent output memory exceed error
            name = 'best' + '.pt'
            torch.save(model, name)
            best_acc = acc_sum / n
            stale = 0
        else:
            stale += 1
            if stale > patience:
                print(
                    f"No improvment {patience} consecutive epochs, early stopping")
                break
    model = torch.load('best' + '.pt')
    save_path = 'result/' + 'result'
    save_path_acc = 'result/' + 'acc'
    compute_metrics(model, save_path, save_path_acc,
                    test_loader, plot_roc_curve=False)


if __name__ == '__main__':
    trainning(n_epochs=4500, patience=4500, batch_size=8 *
              1024, data_path='Homo_sapiens_data.txt')
