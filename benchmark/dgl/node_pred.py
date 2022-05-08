import argparse

from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

import torch
import torch.nn.functional as F

from utils import Logger, EarlyStopping
from model import GNN


def train(model, g, x, y, train_idx, optimizer, task_type):
    model.train()

    optimizer.zero_grad()
    out = model(g, x)
    if task_type == 'binary classification':
        loss = F.binary_cross_entropy_with_logits(
            out[train_idx], y.squeeze(1)[train_idx])
    elif task_type == 'multiclass classification':
        loss = F.cross_entropy(out[train_idx], y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, g, x, y, split_idx, evaluator, eval_metric):
    model.eval()

    out = model(g, x)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_metric = evaluator.eval({
        'y_true': y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })[eval_metric]
    valid_metric = evaluator.eval({
        'y_true': y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })[eval_metric]
    test_metric = evaluator.eval({
        'y_true': y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })[eval_metric]

    return train_metric, valid_metric, test_metric


def run_node_pred(args, model, dataset):
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    model.to(device)

    # dataset = DglNodePropPredDataset(name=args.dataset, root=args.dataset_path)
    evaluator = Evaluator(name=args.dataset)

    g, y = dataset[0]

    # add reverse edges
    srcs, dsts = g.all_edges()
    g.add_edges(dsts, srcs)

    # add self-loop
    print(f'Total edges before adding self-loop {g.number_of_edges()}')
    g = g.remove_self_loop().add_self_loop()
    print(f'Total edges after adding self-loop {g.number_of_edges()}')

    g, y = g.to(device), y.to(device)

    if args.dataset == 'ogbn-proteins':
        x = g.ndata['species']
    else:
        x = g.ndata['feat']

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']

    logger = Logger(args.runs, mode='max')

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        early_stopping = EarlyStopping(
            patience=args.patience, verbose=True, mode='max')

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, g, x, y, train_idx,
                         optimizer, dataset.task_type)
            result = test(model, g, x, y, split_idx,
                          evaluator, dataset.eval_metric)
            logger.add_result(run, result)

            train_acc, valid_acc, test_acc = result

            if epoch % args.log_steps == 0:
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')

            if early_stopping(valid_acc, model):
                break

        logger.print_statistics(run)
    logger.print_statistics()


def main():
    parser = argparse.ArgumentParser(
        description='train node property prediction')
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv',
                        choices=['ogbn-arxiv'])
    parser.add_argument('--dataset_path', type=str, default='/dev/dataset',
                        help='path to dataset')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--model', type=str, default='sage')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=30)
    args = parser.parse_args()
    print(args)

    dataset = DglNodePropPredDataset(name=args.dataset, root=args.dataset_path)
    g, _ = dataset[0]

    num_features = g.ndata['feat'].shape[1]

    model = GNN(num_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers,
                    args.dropout, args.model)

    run_node_pred(args, model, dataset)


if __name__ == '__main__':
    main()
