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
import random

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
parser.add_argument('-run_id', default='pretrained4')
parser.add_argument('-var_x', type=float)

args = parser.parse_args()
print(f"args:{args}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device:{device}")


task = "linear_classification"
run_id = args.run_id  # if you train more models, replace with the run_id from the table above
run_path = os.path.join(run_dir, task, run_id)

print(f"Load model..")
model, conf = get_model_from_run(run_path)
model.to(device)


n_dims = conf.model.n_dims
# batch_size = conf.training.batch_size
batch_size = 1000

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
# def testSample(n_points, b_size, bias1, bias2, std, std_x, p1,p2, frac_pos):
#     x_bias1 = torch.normal(mean=bias1, std=std, size=(1, n_dims))
#     x_bias2 = torch.normal(mean=bias2, std=std, size=(1, n_dims))
#     if n_points>0:
#         ## first obtain in-context examples
#         y = torch.zeros((b_size, n_points), dtype=torch.int32)
#         x = torch.randn(b_size, n_points, n_dims) * std_x
#         split_index = int(n_points * frac_pos)
#         if split_index > 0:
#             y[:, :split_index] = 1
#             pos_pos_num = int(split_index*p1)
#             if pos_pos_num>0:
#                 x[:,:pos_pos_num,:]+=x_bias1
#             if split_index-pos_pos_num>0:
#                 x[:,pos_pos_num:split_index,:]+=x_bias2
#         ## positive class: [positive sample (p1),+1], [negative sampel(1-p1), +1]
#         if n_points-split_index>0:
#             y[:, split_index:] = -1
#             neg_pos_num = int((1-p2)*(n_points-split_index))
#             if neg_pos_num>0:
#                 x[:,split_index:(split_index+neg_pos_num),:]+=x_bias1
#             if n_points-split_index-neg_pos_num>0:
#                 x[:,(split_index+neg_pos_num):, :]+=x_bias2
#         ## negative class: [positive sampel(1-p2), -1], [negative sample(p2), -1]
#         x, y = shuffle_within_batch(x, y)
        
#         ## add query samples
#         split_batch = b_size // 2
#         additional_first_half = torch.randn(split_batch, 1, n_dims) 
#         additional_first_half += x_bias1
#         additional_second_half = torch.randn(b_size - split_batch, 1, n_dims) 
#         additional_second_half += x_bias2
#         additional_points = torch.cat([additional_first_half, additional_second_half], dim=0)
#         x = torch.cat([x, additional_points], dim=1)
        
#         additional_first_half = torch.ones((split_batch, 1))
#         additional_second_half = -torch.ones((b_size - split_batch, 1))
#         additional_labels = torch.cat([additional_first_half, additional_second_half], dim=0)
#         y = torch.cat([y, additional_labels], dim=1)
    
#     else:
#         #zero-shot, only query samples
#         split_batch = b_size // 2
#         additional_first_half = torch.randn(split_batch, 1, n_dims) 
#         additional_first_half += x_bias1
#         additional_second_half = torch.randn(b_size - split_batch, 1, n_dims) 
#         additional_second_half += x_bias2
#         x = torch.cat([additional_first_half, additional_second_half], dim=0)
#         additional_first_half = torch.ones((split_batch, 1))
#         additional_second_half = -torch.ones((b_size - split_batch, 1))
#         y = torch.cat([additional_first_half, additional_second_half], dim=0)
    
#     return x,y

def testSample(n_points, b_size, bias1, bias2, std, std_x, p1,p2, frac_pos, frac_pos_test):
    x_bias1 = torch.normal(mean=bias1, std=std, size=(1, n_dims))
    x_bias2 = torch.normal(mean=bias2, std=std, size=(1, n_dims))
    if n_points>0:
        ## first obtain in-context examples
        y = torch.zeros((b_size, n_points), dtype=torch.int32)
        x = torch.randn(b_size, n_points, n_dims) #* std_x
        split_index = int(n_points * frac_pos)
        if split_index > 0:
            x[:,:split_index,:]+=x_bias1
            pos_pos_num = int(split_index*p1)
            if pos_pos_num>0:
                y[:, :pos_pos_num] = 1
            if split_index-pos_pos_num>0:
                y[:, pos_pos_num:split_index] = -1
        ## positive class: [positive sample,+1(p1)], [positive sampel, -1(1-p1)]
        if n_points-split_index>0:
            x[:,split_index:,:]-=x_bias1
            neg_pos_num = int((1-p2)*(n_points-split_index))
            if neg_pos_num>0:
                y[:, split_index:(split_index+neg_pos_num)] = 1
            if n_points-split_index-neg_pos_num>0:
                y[:, (split_index+neg_pos_num):] = -1
        ## negative class: [positive sampel(1-p2), -1], [negative sample(p2), -1]
        x, y = shuffle_within_batch(x, y)
        
        ## add query samples
        split_batch = int(b_size * frac_pos_test)
        additional_first_half = torch.randn(split_batch, 1, n_dims) 
        additional_first_half += x_bias1
        additional_second_half = torch.randn(b_size - split_batch, 1, n_dims) 
        additional_second_half -= x_bias1
        additional_points = torch.cat([additional_first_half, additional_second_half], dim=0)
        x = torch.cat([x, additional_points], dim=1)
        
        additional_first_half = torch.ones((split_batch, 1))
        additional_second_half = -torch.ones((b_size - split_batch, 1))
        additional_labels = torch.cat([additional_first_half, additional_second_half], dim=0)
        y = torch.cat([y, additional_labels], dim=1)
    
    else:
        #zero-shot, only query samples
        split_batch = int(b_size * frac_pos_test)
        additional_first_half = torch.randn(split_batch, 1, n_dims) 
        additional_first_half += x_bias1
        additional_second_half = torch.randn(b_size - split_batch, 1, n_dims) 
        additional_second_half -= x_bias1
        x = torch.cat([additional_first_half, additional_second_half], dim=0)
        additional_first_half = torch.ones((split_batch, 1))
        additional_second_half = -torch.ones((b_size - split_batch, 1))
        y = torch.cat([additional_first_half, additional_second_half], dim=0)
    
    return x,y
    
    


def testF(n_points, b_size, bias1, bias2, std, std_x, p1, p2, frac_pos, frac_pos_test, model, rep=1000):

    random.seed(10)
    pos_pos_ls = []
    neg_neg_ls = []
    
    for i in range(rep):
        # print(f"Rep:{i}")
        xs, ys = testSample(n_points, b_size, bias1, bias2, std,std_x, p1, p2, frac_pos, frac_pos_test)
        xs, ys = xs.to(device), ys.to(device)
        split_batch = int(b_size * frac_pos_test)
        
        # print(f"xs:{xs}")
        # print(f"ys:{ys}")
        # exit(0)
        
        
        with torch.no_grad():
            pred = model(xs, ys).sign()
            pred = pred.cpu()

        pred_pos = pred[:split_batch,-1]
        pred_neg = pred[split_batch:,-1]

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

std_x = args.var_x ** 0.5

# p1_list = [1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0]
# p2_list = [1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0]
# p1_list=[0.0]
# p2_list = [0.0]

frac_list = [0.95, 0.9, 0.8, 0.75, 0.25, 0.2, 0.1, 0.05]
# frac_list = [0.5]
# k = [0,1,2,3,4,5,6,7,8,9,10,20,50,80,100]

# p1_list = [1.0,0.9]
# p2_list = [1.0,0.9]

# df_pos = pd.DataFrame(index=p1_list, columns=p2_list)
# df_neg = pd.DataFrame(index=p1_list, columns=p2_list)
df = pd.DataFrame(index=frac_list, columns=['positive', 'negative'])


# for p1 in p1_list:
#     for p2 in p2_list:
#         dist_pos, dist_neg, mar = testF(n_points=args.n_points, b_size=batch_size, bias1=args.bias1, bias2=args.bias2,std_x=std_x, std=args.std, p1=p1, p2=p2, frac_pos=args.frac_pos, model=model, rep=100)
#         print(f"p1:{p1},p2:{p2},dist_pos:{dist_pos},dist_neg:{dist_neg},CI margin:{mar}")
#         df_pos.at[p1,p2]=dist_pos[0]
#         df_neg.at[p1,p2]=dist_neg[1]

for frac in frac_list:
    dist_pos, dist_neg, mar = testF(n_points=args.n_points, b_size=batch_size, bias1=args.bias1, bias2=args.bias2, std=args.std,std_x=std_x, p1=1.0, p2=1.0, frac_pos=frac, frac_pos_test=0.95, model=model, rep=100)
    print(f"frac:{frac}, dist_pos:{dist_pos},dist_neg:{dist_neg},CI margin:{mar}")
    df.at[frac,'positive']=dist_pos[0]
    df.at[frac,'negative']=dist_neg[1]

# for n_points in k:
#     dist_pos, dist_neg, mar = testF(n_points=n_points, b_size=batch_size, bias1=args.bias1, bias2=args.bias2, std=args.std, std_x=std_x, p1=1.0, p2=1.0, frac_pos=0.5, model=model, rep=100)
#     print(f"n_points:{n_points}, dist_pos:{dist_pos},dist_neg:{dist_neg},CI margin:{mar}")
#     # exit(0)
#     df.at[n_points,'positive']=dist_pos[0]
#     df.at[n_points,'negative']=dist_neg[1]

    

save_dir = "results/"+run_id
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
# df_pos.to_csv(os.path.join(save_dir, "labelnoise0.5-0.5_"+"frac"+str(args.frac_pos)+"_fewshot"+str(args.n_points)+"_pos3.csv"))
# df_neg.to_csv(os.path.join(save_dir, "labelnoise0.5-0.5_"+"frac"+str(args.frac_pos)+"_fewshot"+str(args.n_points)+"_neg3.csv"))
# df.to_csv(os.path.join(save_dir, "early0.5-0.5_"+"_varX"+str(args.var_x)+".csv"))
# df_neg.to_csv(os.path.join(save_dir, "contradtci-0.50.5_"+"_fewshot"+str(args.n_points)+"_neg.csv"))
df.to_csv(os.path.join(save_dir, "extremeFrac"+".csv"))

print('Done.')

        

# dist_pos, dist_neg, mar = testF(n_points=args.n_points, b_size=batch_size, bias1=args.bias1, bias2=args.bias2, std=args.std, p1=args.p1, p2=args.p2, frac_pos=args.frac_pos, model=model, rep=100)
# print(f"dist_pos:{dist_pos}")
# print(f"dist_neg:{dist_neg}")
# print(f"CI margin:{mar}")
# print(f"Done.")