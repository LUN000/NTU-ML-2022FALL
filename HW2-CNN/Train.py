from data import P1dataset
from Model import *
import argparse
import numpy as np
import random
import math
import json
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import os
from PIL import Image

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", "-lr", default=0.00001,type=float, help="learning rate")
    parser.add_argument("--batch", "-bt", default=64 , type=int, help="training batch")
    parser.add_argument("--epoch", "-e", default=800 , type=int, help="training epoch")
    parser.add_argument("--model_path", "-model" , help="pretrain model path")
    parser.add_argument("--vmodel_path", "-vmodel", default=f"/acc_bestmodel.pt" , help="validate model path")
    parser.add_argument("--model_state_dict", "-model_state", help="pretrain model state dict path")
    parser.add_argument("--early_stop", "-es", default=50, help="stop after n epoch valid loss not down")
    parser.add_argument("--data_path", help="training data folder")
    parser.add_argument("--log_out_path", "-lop")
    parser.add_argument("--model_out_path", "-mop")
    args = parser.parse_args()

    args.log_out_path = os.path.join(os.path.abspath(os.getcwd()), args.log_out_path)
    args.model_out_path = os.path.join(os.path.abspath(os.getcwd()), args.model_out_path)

    if args.log_out_path is not None:
        os.makedirs(args.log_out_path, exist_ok=True)
    if args.model_out_path is not None:
        os.makedirs(args.model_out_path, exist_ok=True)

    return args

def P1train(args, train_dataloader, validate_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} now")
    
    if args.model_path:
        Model = torch.load(args.model_path).to(device)
        print(f"Training pretrain model {args.model_path}!\n{args}\n{Model.__name__}")
    elif args.model_state_dict:
        Model = P1Cnn_b().to(device)
        Model.load_state_dict(torch.load(args.model_pypstate_dict))
        print(f"Training pretrain model {args.model_state_dict}!\n{args}\n{Model.__name__}")
    else:
        Model = ResNet152().to(device)
        print(f"Train from scratch!\n{args}\n{Model.__name__}")
    
    # weight = torch.Tensor([6525/3579, 6525/402, 6525/3736, 1, 6525/4321, 6525/2872, 6525/4452])
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(Model.parameters(), lr = args.learning_rate)
    loss_min, acc_min, early_stop = 1.8, 0.35, 0

    tr_loss, v_loss, acc = {}, {}, {}
    
    for epoch in range(args.epoch):
        Model.train()
        loss_total, predict = 0, 0
        
        for i, (X, y) in enumerate(train_dataloader):
            optimizer.zero_grad()

            X, y = X.float().to(device), y.to(device)
            y_hat = Model(X)
            y_hat_indices = y_hat.max(axis=1).indices
            predict += len(y[y==y_hat_indices])

            loss = criterion(y_hat, y)
            loss.backward()
            loss_total = loss_total + loss
            
            torch.autograd.set_detect_anomaly(True)
            optimizer.step()
        
        # accuracy = predict/((i+1)*args.batch)
        loss_ave = (loss_total / (i+1)).tolist()
        loss_valid, accuracy = P1validate(args, validate_dataloader, Model)
        tr_loss.update({epoch : loss_ave})
        v_loss.update({epoch : loss_valid})
        acc.update({epoch : accuracy})
        
        if (epoch % 1 == 0):
            torch.cuda.empty_cache() # release unuse tensor
            print('epoch = %5d, train_loss = %3.5f, valid_loss = %3.5f, acc = %1.5f' % (epoch, loss_ave, loss_valid, accuracy))

        # early stop
        if loss_valid < loss_min:
            loss_min = loss_valid
            early_stop = 0
        else:
            early_stop += 1
            if early_stop == int(args.early_stop):
                print("early stop! epoch = %5d, train_loss = %3.5f, valid_loss = %3.5f, acc = %1.5f" % (epoch, loss_ave, loss_valid, accuracy))
                break

        #儲存最佳loss model
        if accuracy > acc_min:
            acc_min = accuracy
            torch.save(Model, f'{args.model_out_path}/acc_bestmodel.pt')
            print(
                'save best acc model at epoch = %5d, train_loss = %3.5f, valid_loss = %3.5f, acc = %1.5f' % (epoch, loss_ave, loss_valid, accuracy)
                )
            
            with open(f"{args.log_out_path}/train_loss.json", "w") as file:
                file.write(json.dumps(tr_loss))
            file.close()
            with open(f"{args.log_out_path}/valid_loss.json", "w") as file:
                file.write(json.dumps(v_loss))
            file.close()
            with open(f"{args.log_out_path}/acc.json", "w") as file:
                file.write(json.dumps(acc))
            file.close()

    with open(f"{args.log_out_path}/train_loss.json", "w") as file:
        file.write(json.dumps(tr_loss))
    file.close()
    with open(f"{args.log_out_path}/valid_loss.json", "w") as file:
        file.write(json.dumps(v_loss))
    file.close()
    with open(f"{args.log_out_path}/acc.json", "w") as file:
        file.write(json.dumps(acc))
    file.close()
    print(f"Finish Training {args.epoch}")

@torch.no_grad()
def P1validate(args, validate_dataloader, model, state='training_validate'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if state == "final_validate":
        try:
            Model = torch.load(args.vmodel_path).to(device)
        except:
            Model = Cnn_b().to(device)
            Model.load_state_dict(torch.load(args.vmodel_path))
    else:
        Model = model.to(device)
    class_criterion = nn.CrossEntropyLoss().to(device)

    Model.eval()
    loss_total, predict = 0, 0
    for i, (X, y) in enumerate(validate_dataloader):
        X, y = X.float().to(device), y.to(device)
        y_hat = Model(X)
        y_hat_indices = y_hat.max(axis=1).indices
        predict += len(y[y==y_hat_indices])
        loss = class_criterion(y_hat, y)
        loss_total += float(loss)
    
    loss_ave = loss_total/(i+1)
    accuracy = predict/((i+1)*args.batch)
    if state == "final_validate":
        print(f"validate_loss: {loss_ave}")
    return [loss_ave, accuracy]


if __name__ == "__main__":
    setup_seed(999)
    args = arg_parser()

    transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.RandomHorizontalFlip(p=0.5)
                                ])
    print(transform)

    all_train_dataset = P1dataset(f"{args.data_path}/train.csv", f"{args.data_path}/train", transform)
    train_set_size = int(len(all_train_dataset) * 0.8)
    valid_set_size = len(all_train_dataset) - train_set_size
    train_dataset, valid_dataset = random_split(all_train_dataset, [train_set_size, valid_set_size])
    train_dataloader, valid_dataloader = DataLoader(train_dataset, args.batch, num_workers=5, shuffle=True), DataLoader(valid_dataset, args.batch, num_workers=5)

    P1train(args, train_dataloader, valid_dataloader)
