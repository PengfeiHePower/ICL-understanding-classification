# zero-shot
python infer.py -n_points 0 -bias1 1 -bias2 -1 -std 1.0 -p1 1 -p2 1 -frac_pos 1.0 

# 100-shot
python infer.py -n_points 100 -bias1 1 -bias2 -1 -std 1.0 -p1 1 -p2 1 -frac_pos 1.0
CUDA_VISIBLE_DEVICE=1 python infer.py -n_points 100 -bias1 1 -bias2 -1 -std 1.0 -p1 1.0 -p2 0.1 -frac_pos 0
CUDA_VISIBLE_DEVICE=1 python infer.py -n_points 100 -bias1 1 -bias2 -1 -std 1.0 -p1 1.0 -p2 0.4 -frac_pos 0
CUDA_VISIBLE_DEVICE=1 python infer.py -n_points 100 -bias1 0.5 -bias2 -1 -std 1.0 -p1 0.95 -p2 1.0 -frac_pos 1
CUDA_VISIBLE_DEVICE=1 python infer.py -n_points 100 -bias1 0.5 -bias2 -1 -std 1.0 -p1 0.93 -p2 1.0 -frac_pos 1
CUDA_VISIBLE_DEVICE=1 python infer.py -n_points 100 -bias1 0.5 -bias2 -1 -std 1.0 -p1 0.97 -p2 1.0 -frac_pos 1
CUDA_VISIBLE_DEVICE=1 python infer.py -n_points 100 -bias1 0.5 -bias2 -1 -std 1.0 -p1 0.92 -p2 1.0 -frac_pos 1

python infer.py -n_points 100 -bias1 0.5 -bias2 -0.5 -std 1.0 -p1 1.0 -p2 0.85 -frac_pos 0
CUDA_VISIBLE_DEVICE=1 python infer.py -n_points 100 -bias1 0.5 -bias2 -0.5 -std 1.0 -p1 1.0 -p2 0.1 -frac_pos 0

CUDA_VISIBLE_DEVICE=1 python infer.py -n_points 100 -bias1 0.5 -bias2 -0.5 -std 2.0 -p1 0.4 -p2 1.0 -frac_pos 1

CUDA_VISIBLE_DEVICE=1 python infer.py -n_points 100 -bias1 0.5 -bias2 -0.5 -std 2.0 -p1 1.0 -p2 0 -frac_pos 0

CUDA_VISIBLE_DEVICE=1 python infer.py -n_points 100 -bias1 2 -bias2 -0 -std 1 -p1 1.0 -p2 0 -frac_pos 0

CUDA_VISIBLE_DEVICE=1 python infer.py -n_points 100 -bias1 2 -bias2 1 -std 1 -p1 1.0 -p2 1.0 -frac_pos 0

CUDA_VISIBLE_DEVICE=1 python infer.py -n_points 100 -bias1 -1 -bias2 1 -std 1 -p1 1.0 -p2 1.0 -frac_pos 0.9
