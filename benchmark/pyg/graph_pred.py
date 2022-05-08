import argparse

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from utils import Logger, EarlyStopping
from model import EGNN

from tqdm.auto import tqdm


def train(model, device, loader, optimizer):
    model.train()

    total_loss = 0
    total = 0

    for batch in tqdm(loader, desc='Train'):
        batch = batch.to(device)

        yh = model(batch)
        optimizer.zero_grad()

        y = batch.y

        # loss = F.mse_loss(yh.float(), y.float())
        loss = F.binary_cross_entropy_with_logits(yh.float(), y.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total += 1

    return total_loss / total


@torch.no_grad()
def eval(model, device, loader, evaluator, eval_metric):
    model.eval()
    y_true = []
    y_pred = []

    for batch in tqdm(loader, desc='Eval '):
        batch = batch.to(device)

        yh = model(batch)

        y = batch.y

        y_true.append(y.view(yh.shape).detach().cpu())
        y_pred.append(yh.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {'y_true': y_true, 'y_pred': y_pred}

    return evaluator.eval(input_dict)[eval_metric]


def run_graph_pred(args, model, dataset):
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    model.to(device)

    # dataset = DglNodePropPredDataset(name=args.dataset, root=args.dataset_path)
    evaluator = Evaluator(name=args.dataset)

    split_idx = dataset.get_idx_split()

    train_loader = DataLoader(dataset[split_idx['train']], batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(dataset[split_idx['valid']], batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(dataset[split_idx['test']], batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    logger = Logger(args.runs, mode='max')

    for run in range(args.runs):
        print('\nRun {}'.format(run + 1))
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        early_stopping = EarlyStopping(
            patience=args.patience, verbose=True, mode='max')

        for epoch in range(1, 1 + args.epochs):
            print('epoch {}'.format(epoch))
            loss = train(model, device, train_loader, optimizer)

            train_metric = eval(model, device, train_loader,
                                evaluator, dataset.eval_metric)
            valid_metric = eval(model, device, valid_loader,
                                evaluator, dataset.eval_metric)
            test_metric = eval(model, device, test_loader,
                               evaluator, dataset.eval_metric)

            result = [train_metric, valid_metric, test_metric]

            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                print(
                      f'Loss: {loss:.4f}, '
                      f'Train: {train_metric:.4f}, '
                      f'Valid: {valid_metric:.4f} '
                      f'Test: {test_metric:.4f}')
                print()

            if early_stopping(valid_metric, model):
                break

        logger.print_statistics(run)
    logger.print_statistics()


def main():
    parser = argparse.ArgumentParser(
        description='train graph property prediction')
    parser.add_argument('--dataset', type=str, default='ogbg-molhiv',
                        choices=['ogbg-molhiv'])
    parser.add_argument('--dataset_path', type=str, default='/dev/dataset',
                        help='path to dataset')
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--hidden_channels', type=int, default=300)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--model', type=str, default='gin')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=30)
    args = parser.parse_args()
    print(args)

    dataset = PygGraphPropPredDataset(
        name=args.dataset, root=args.dataset_path)

    model = EGNN(args.hidden_channels,
                     dataset.num_tasks, args.num_layers,
                     args.dropout, args.model)

    run_graph_pred(args, model, dataset)


if __name__ == '__main__':
    main()
