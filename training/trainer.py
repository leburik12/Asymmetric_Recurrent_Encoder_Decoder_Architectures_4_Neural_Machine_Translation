class Trainer:
    def __init__(self, model, optimizer, loss_fn, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train_epoch(self, data_iter):
        self.model.train()
        total_loss = 0

        for batch in data_iter:
            src, tgt, src_len, _ = [x.to(self.device) for x in batch]

            self.optimizer.zero_grad()
            output = self.model(src, tgt[:, :-1], src_len)

            loss = self.loss_fn(
                output.reshape(-1, output.shape[-1]),
                tgt[:, 1:].reshape(-1)
            )

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(data_iter)

