from collections import OrderedDict
import re
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm.notebook import tqdm
import numpy as np

from eval import get_run_metrics, read_run_dir, get_model_from_run
from plot_utils import basic_plot, collect_results, relevant_model_names
from tasks import get_task_sampler

from scipy import stats

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

sns.set_theme('notebook', 'darkgrid')
palette = sns.color_palette('colorblind')

run_dir = "../models"

#args.n_points, args.bias1, args.bias2, args.std, args.p1, args.p2, args.frac_pos
import argparse
parser = argparse.ArgumentParser(description="ICL infer.")
parser.add_argument('-n_points', type=int)
parser.add_argument('-bias1', type=float)
parser.add_argument('-bias2', type=float)
parser.add_argument('-std', type=float)
parser.add_argument('-p1', type=float)
parser.add_argument('-p2', type=float)
parser.add_argument('-frac_pos', type=float)

args = parser.parse_args()
print(f"args:{args}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device:{device}")


task = "linear_classification"
run_id = "pretrained1"  # if you train more models, replace with the run_id from the table above
run_path = os.path.join(run_dir, task, run_id)

print(f"Load model..")
model, conf = get_model_from_run(run_path)
model.to(device)


n_dims = conf.model.n_dims
batch_size = conf.training.batch_size

# data_sampler = get_data_sampler(conf.training.data, n_dims)
task_sampler = get_task_sampler(
    conf.training.task,
    n_dims,
    batch_size,
    **conf.training.task_kwargs
)

task = task_sampler()

def shuffle_within_batch(xs, ys):
    b_size, n_points, _ = xs.shape
    # Generate a random permutation for each batch
    permutations = torch.randperm(n_points).repeat(b_size, 1)

    # Apply the same permutation to each batch in xs and ys
    xs_shuffled = torch.gather(xs, 1, permutations.unsqueeze(-1).expand(-1, -1, xs.size(2)))
    ys_shuffled = torch.gather(ys, 1, permutations)

    return xs_shuffled, ys_shuffled

## generate inference data
def testSample(n_points, b_size, bias1, bias2, std, p1,p2,frac_pos):
    x_bias1 = torch.normal(mean=bias1, std=std, size=(1, n_dims))
    x_bias2 = torch.normal(mean=bias2, std=std, size=(1, n_dims))
    split_index = int(n_points * frac_pos)
    if n_points>0:
        ## first obtain in-context examples
        xs_b = torch.randn(b_size, n_points, n_dims)
        xs_b[:, :split_index, :] += x_bias1
        xs_b[:, split_index:, :] += x_bias2
        
        true_y = torch.empty(b_size, n_points)
        probs_first_half = torch.tensor([p1, 1-p1])  # Probability for 1 and -1 respectively
        choices_first_half = torch.tensor([1, -1])
        if split_index>0:
            first_half = torch.multinomial(probs_first_half, b_size * split_index, replacement=True).reshape(b_size, split_index)
            true_y[:, :split_index] = choices_first_half[first_half]
        probs_second_half = torch.tensor([1-p2, p2])  # Probability for 1 and -1 respectively
        choices_second_half = torch.tensor([1, -1])
        if n_points - split_index>0:
            second_half = torch.multinomial(probs_second_half, b_size * (n_points - split_index), replacement=True).reshape(b_size, n_points - split_index)
            true_y[:, split_index:] = choices_second_half[second_half]
        #shuffle examples
        xs_b, true_y = shuffle_within_batch(xs_b, true_y)
        
        ## add query samples
        split_batch = b_size // 2
        additional_first_half = torch.randn(split_batch, 1, n_dims) 
        additional_first_half += x_bias1
        additional_second_half = torch.randn(b_size - split_batch, 1, n_dims) 
        additional_second_half += x_bias2
        additional_points = torch.cat([additional_first_half, additional_second_half], dim=0)
        xs_b = torch.cat([xs_b, additional_points], dim=1)
        
        additional_first_half = torch.ones((split_batch, 1))
        additional_second_half = -torch.ones((b_size - split_batch, 1))
        additional_labels = torch.cat([additional_first_half, additional_second_half], dim=0)
        true_y = torch.cat([true_y, additional_labels], dim=1)
    
    else:
        #zero-shot, only query samples
        split_batch = b_size // 2
        additional_first_half = torch.randn(split_batch, 1, n_dims) 
        additional_first_half += x_bias1
        additional_second_half = torch.randn(b_size - split_batch, 1, n_dims) 
        additional_second_half += x_bias2
        xs_b = torch.cat([additional_first_half, additional_second_half], dim=0)
        additional_first_half = torch.ones((split_batch, 1))
        additional_second_half = -torch.ones((b_size - split_batch, 1))
        true_y = torch.cat([additional_first_half, additional_second_half], dim=0)
    
    return xs_b, true_y


def testF(n_points, b_size, bias1, bias2, std, p1, p2, frac_pos, model, rep=100):
    pos_pos_ls = []
    neg_neg_ls = []
    
    for i in range(rep):
        # print(f"Rep:{i}")
        xs, ys = testSample(n_points, b_size, bias1, bias2, std, p1, p2, frac_pos)
        xs, ys = xs.to(device), ys.to(device)
        
        with torch.no_grad():
            pred = model(xs, ys).sign()
            pred = pred.cpu()

        pred_pos = pred[:b_size//2,-1]
        pred_neg = pred[b_size//2:,-1]

        pos_pos = sum(pred_pos==1)/len(pred_pos)
        neg_neg = sum(pred_neg==-1)/len(pred_neg)
        
        pos_pos_ls.append(pos_pos)
        neg_neg_ls.append(neg_neg)
    
    pos_pos_np = np.array(pos_pos_ls)
    neg_neg_np = np.array(neg_neg_ls)
    
    pos_pos_avg = np.mean(pos_pos_np)
    neg_neg_avg = np.mean(neg_neg_np)
    
    pos_pos_std = np.std(pos_pos_np, ddof=1)
    neg_neg_std = np.std(neg_neg_np, ddof=1)
    
    n = len(pos_pos_ls)
    confidence_level = 0.95
    degrees_of_freedom = n - 1
    alpha = 1 - confidence_level
    t_critical = stats.t.ppf(1 - alpha/2, degrees_of_freedom)
    pos_pos_mar = t_critical * (pos_pos_std / np.sqrt(n))
    neg_neg_mar = t_critical * (neg_neg_std / np.sqrt(n))
    
    return [pos_pos_avg, 1-pos_pos_avg], [1-neg_neg_avg, neg_neg_avg], [pos_pos_mar, neg_neg_mar]

dist_pos, dist_neg, mar = testF(n_points=args.n_points, b_size=batch_size, bias1=args.bias1, bias2=args.bias2, std=args.std, p1=args.p1, p2=args.p2, frac_pos=args.frac_pos, model=model, rep=100)
print(f"dist_pos:{dist_pos}")
print(f"dist_neg:{dist_neg}")
print(f"CI margin:{mar}")
print(f"Done.")