import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        learning_rate=0.001,
        device="cpu"
    ):

        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate
        )

    def train_one_epoch(self):

        self.model.train()

        total_loss = 0

        loop = tqdm(
            self.train_loader,
            desc="Training"
        )

        for X_batch, y_batch in loop:

            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            outputs = self.model(X_batch)

            loss = self.criterion(
                outputs,
                y_batch
            )

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

            loop.set_postfix(
                loss=loss.item()
            )

        avg_loss = total_loss / len(
            self.train_loader
        )

        return avg_loss