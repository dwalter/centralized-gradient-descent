import torch.nn.functional as F
from optimize import Optimizer


def train(args, model, device, train_loader, epoch):
    model.train()

    optimizer = Optimizer(model)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        output, out_fc0 = model(data)
        outputs = [out_fc0, output]
        loss = F.nll_loss(output, target)

        optimizer.step(loss, outputs)

        if batch_idx % args.log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({format(100.0 * batch_idx / len(train_loader), '.1f')}%)]\tLoss: {format(loss.item(), '.6f')}"
            )
            if args.dry_run:
                break
