import pandas as pd
import numpy as np
import spektral
import torch
from torch import nn
from torch.optim import Adam


def masked_softmax_cross_entropy(logits, labels, mask):
    loss = nn.CrossEntropyLoss(reduction='none')(logits, torch.argmax(labels, dim=1))
    mask = mask.type(torch.float32)
    mask /= torch.mean(mask)
    loss *= mask
    return torch.mean(loss)


def masked_accuracy(logits, labels, mask):
    correct_prediction = (torch.argmax(logits, 1) == torch.argmax(labels, 1))
    accuracy_all = correct_prediction.type(torch.float32)
    mask = mask.type(torch.float32)
    mask /= torch.mean(mask)
    accuracy_all *= mask

    return torch.mean(accuracy_all)


class CoraGnn(nn.Module):
    def __init__(self, feat_len, units, lbl_len):
        super(CoraGnn, self).__init__()
        self.lyr_1 = nn.Linear(feat_len, units)
        self.lyr_2 = nn.Linear(units, lbl_len)
        self.relu = nn.ReLU()

    def forward(self, input_x, adj):
        x = self.lyr_1(input_x)
        x = torch.matmul(adj, x)
        x = self.relu(x)
        x = torch.matmul(adj, x)
        out = self.lyr_2(x)

        return out


def train_cora(fts, adj, units, epochs, lr):
    net = CoraGnn(len(fts[0]), units, len(labels[0]))
    optimizer = Adam(net.parameters(), lr=lr)

    best_accuracy = 0.0

    for ep in range(epochs + 1):
        logits = net(fts, adj)
        loss = masked_softmax_cross_entropy(logits, labels, train_mask)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            logits = net(fts, adj)
            val_accuracy = masked_accuracy(logits, labels, val_mask)
            test_accuracy = masked_accuracy(logits, labels, val_mask)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            print("Epoch: {} | Train Loss: {} | Val Acc: {} | Test Acc: {}".format(ep, loss.item(), val_accuracy.item(),
                                                                                   test_accuracy.item()))


def main():
    global adj, labels, train_mask, val_mask, test_mask
    dataset = spektral.datasets.citation.Citation('cora')
    graph = dataset.graphs[0]
    #   a: the adjacency matrix
    #   x: the node features
    #   e: the edge features
    #   y: the labels

    adj = torch.from_numpy(graph.a.todense())
    features = torch.from_numpy(graph.x)
    labels = torch.from_numpy(graph.y)
    train_mask = torch.from_numpy(dataset.mask_tr)
    val_mask = torch.from_numpy(dataset.mask_va)
    test_mask = torch.from_numpy(dataset.mask_te)

    adj = adj + torch.eye(adj.shape[0])
    features = features
    # adj = adj.type(torch.float32)

    print("Details:")
    print("Feat. #:", features.shape[1])
    print("Adj. matrix:", adj.shape)
    print("Class #:", labels.shape[1])

    print("Train #:\t", torch.sum(train_mask).item())
    print("Val #:\t", torch.sum(val_mask).item())
    print("Test #:\t", torch.sum(test_mask).item())
    print("+++++++++++++++++++++++")

    lr = 1e-2
    epochs = 200
    units = 32
    print("Regular neighbor summation")
    train_cora(features, adj, units, epochs, lr)
    print()

    print("Normalized neighbor summation")
    deg = torch.sum(adj, dim=1)
    train_cora(features, adj / deg, units, epochs, lr)
    print()

    print("Mutual Normalized neighbor summation")
    norm_deg = torch.diag(1.0 / torch.sqrt(deg))
    norm_adj = torch.matmul(norm_deg, torch.matmul(adj, norm_deg))
    train_cora(features, norm_adj, units, epochs, lr)


if __name__ == '__main__':
    main()
