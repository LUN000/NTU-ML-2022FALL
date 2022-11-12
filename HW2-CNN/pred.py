import torch
import argparse
import os
import numpy as np
import random
from data import P1valid_dataset
from torch.utils.data import DataLoader
import pandas as pd

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

@torch.no_grad()
def predict(args, validate_dataloader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device="cpu"
    model = torch.load(args.model, map_location=device).to(device)
    model.eval()

    output = {"id":[],"label":[]}

    for i, (X, names) in enumerate(validate_dataloader):
        X = X.float().to(device)
        pred = model(X)
        pred = pred.cpu()
        labels = pred.max(axis=1).indices.numpy()
        for name, label in zip(names, labels):
            output["id"].append(name)
            output["label"].append(label)

    output = pd.DataFrame.from_dict(output)
    output.to_csv(os.path.join(os.getcwd(), args.outpath, "pred.csv"), index=False)

if __name__=="__main__":
    setup_seed(999)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="acc_bestmodel.pt", help="pretrain model path")
    parser.add_argument("--inpath", help="input image path")
    parser.add_argument("--outpath", help="output csv path")
    
    args = parser.parse_args()
    if os.path.join(os.getcwd(), args.outpath) is not None:
        os.makedirs(args.outpath, exist_ok=True)

    test_dataset =  P1valid_dataset(img_dir=f"{args.inpath}")
    test_dataloader = DataLoader(test_dataset, 64)
    predict(args, test_dataloader)
