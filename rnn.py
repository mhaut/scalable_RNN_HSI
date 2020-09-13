import argparse
import sys
import numpy as np
import random
from sru import *
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import aux


class Model(nn.Module):
    def __init__(self, args, input_size, nclasses=2):
        super(Model, self).__init__()
        if args.vanillarnn: net = nn.RNN
        elif args.lstm: net = nn.LSTM
        elif args.gru: net = nn.GRU
        else: net = SRU
        d_out = args.d

        self.recurrent = net(input_size, args.d, args.depth)#, dropout = args.dropout)
        feat_size = d_out * args.numseq
        self.bn = nn.BatchNorm1d(feat_size)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(feat_size, nclasses)

    def forward(self, x):
        x = self.recurrent(x)[0]
        x = x.permute(1, 0, 2).contiguous()
        x = x.view(x.size(0), -1)
        x = self.tanh(self.bn(x))
        x = self.fc(x)
        return x



def eval_model(model, data_loader, use_cuda):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    predictions = []; real = []
    for _, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = Variable(x)
            if use_cuda: x = x.cuda()

        x = torch.transpose(x, 0, 1).contiguous()
        y = Variable(y)
        if use_cuda: y = y.cuda()
        
        output = model(x)
        loss = criterion(output, y)
        total_loss += loss.item()*x.size(1)
        pred = output.data.max(1)[1]
        predictions = [a for a in pred.data.cpu().numpy()]
        real = [a for a in y.data.cpu().numpy()]

    return loss.item(), (np.array(real), np.array(predictions))



def train_model(model, optimizer, train_loader, use_cuda):
    model.train()
    criterion = nn.CrossEntropyLoss()
    all_loss = 0
    for _, (x, y) in enumerate(train_loader):
        model.zero_grad()
        x = Variable(x)
        if use_cuda: x = x.cuda()
        x = torch.transpose(x, 0, 1).contiguous()
        y = Variable(y)
        if use_cuda: y = y.cuda()
        output = model(x)
        loss = criterion(output, y)
        all_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    return model, optimizer








if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--cnn", action='store_true', help="whether to use cnn")

    argparser.add_argument("--lstm", action='store_true', help="whether to use lstm")
    argparser.add_argument("--gru", action='store_true', help="whether to use gru")
    argparser.add_argument("--vanillarnn", action='store_true', help="whether to use vanilla rnn")
    

    argparser.add_argument("--cudnn", action='store_true', help="whether to use cuda")
    argparser.add_argument("--use_cuda", action='store_false', help="cuda or not")

    argparser.add_argument("--dataset", type=str, default="indian", help="which dataset")
    argparser.add_argument("--numseq", type=int, default=-1) # if numseq -1 numseq = numbands
    argparser.add_argument("--pca", type=int, default=-1) # if numseq -1 numseq = numbands

    argparser.add_argument("--use_val", action='store_true', help="validation or not")
    argparser.add_argument("--valpercent", type=float, default=0.1)

    argparser.add_argument("--random_state", type=int, default=69)
    argparser.add_argument("--batch_size", type=int, default=100)
    argparser.add_argument("--epochs", type=int, default=200)
    argparser.add_argument("--idtest", type=int, default=0)
    argparser.add_argument("--d", type=int, default=64)
    #argparser.add_argument("--dropout", type=float, default=0.10)
    argparser.add_argument("--tpercent", type=float, default=-1) # if -1, 15 indian 10 the others
    argparser.add_argument("--depth", type=int, default=1)
    argparser.add_argument("--lr", type=float, default=0.001)

    
    args = argparser.parse_args()


    torch.backends.cudnn.enabled = True if args.cudnn else False
    if args.tpercent == -1: args.tpercent = 0.15 if args.dataset == "indian" else 0.10

    train_loader, test_loader, val_loader, all_loader, nclasses, bands = aux.get_loaders(args)
    

    model = Model(args, bands, nclasses)
    
    print("PARAMS", sum(p.numel() for p in model.parameters() if p.requires_grad))

    if args.use_cuda: model = model.cuda()
    need_grad = lambda x: x.requires_grad
    optimizer = optim.Adam(
        filter(need_grad, model.parameters()),
        lr = args.lr
    )

    best_acc = -1e+8

    val_loader = val_loader if args.use_val else test_loader
    for epoch in range(args.epochs):
        model, optimizer = train_model(model, optimizer, train_loader, args.use_cuda)
        losstr, (realtr,predstr) = eval_model(model, train_loader, args.use_cuda)
        losste, (realte,predste) = eval_model(model, val_loader, args.use_cuda)
        resultstr = aux.reports(predstr, realtr, range(nclasses))[2]
        resultste = aux.reports(predste, realte, range(nclasses))[2]
        print(epoch, "TRAIN LOSS", losstr, "TRAIN ACC", resultstr[0],\
                           "LOSS", losste, "ACC", resultste[0])

        if resultste[0] > best_acc:
            best_acc = resultste[0]
            torch.save(model.state_dict(), '/tmp/best_model.pth.tar')

    model.load_state_dict(torch.load('/tmp/best_model.pth.tar'))
    _, (real,preds) = eval_model(model, test_loader, args)
    results = aux.reports(preds, real, range(nclasses))[2]
    print(results)
