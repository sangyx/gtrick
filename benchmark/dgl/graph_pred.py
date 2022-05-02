import argparse

from ogb.graphproppred import DglGraphPropPredDataset, Evaluator, collate_dgl

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import Logger, EarlyStopping
from model import EGCN, EGIN

from tqdm.auto import tqdm


def train(model, device, loader, optimizer):
    model.train()

    total_loss = 0
    total = 0

    for batch in tqdm(loader, desc='Train'):
        g, y = batch
        g, y = g.to(device), y.to(device)

        yh = model(g, g.ndata['feat'], g.edata['feat'])
        optimizer.zero_grad()

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
        g, y = batch
        g, y = g.to(device), y.to(device)

        yh = model(g, g.ndata['feat'], g.edata['feat'])

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

    # add self-loop
    for i in range(len(dataset)):
        dataset.graphs[i] = dataset.graphs[i].remove_self_loop(
        ).add_self_loop()

    evaluator = Evaluator(name=args.dataset)

    split_idx = dataset.get_idx_split()

    train_loader = DataLoader(dataset[split_idx['train']], batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, collate_fn=collate_dgl)
    valid_loader = DataLoader(dataset[split_idx['valid']], batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, collate_fn=collate_dgl)
    test_loader = DataLoader(dataset[split_idx['test']], batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers, collate_fn=collate_dgl)

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
    parser.add_argument('--dataset_path', type=str, default='/home/ubuntu/.dgl_dataset',
                        help='path to dataset')
    parser.add_argument('--device', type=int, default=2)
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

    dataset = DglGraphPropPredDataset(
        name=args.dataset, root=args.dataset_path)

    if args.model == 'gin':
        model = EGIN(args.hidden_channels,
                     dataset.num_tasks, args.num_layers,
                     args.dropout)
    elif args.model == 'gcn':
        model = EGCN(args.hidden_channels, dataset.num_tasks,
                     args.num_layers, args.dropout)

    run_graph_pred(args, model, dataset)


if __name__ == '__main__':
    main()
