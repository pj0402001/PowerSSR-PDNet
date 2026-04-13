import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

ROOT = Path(r"D:\安全域\PowerSSR-DL")

def extract_bsp_midpoints(X_safe, X_unsafe, d_threshold):
    """
    使用 KDTree 提取边界样本对 (BSP) 的中点
    """
    print(f"Building KDTree for {len(X_unsafe)} unsafe points...")
    tree_unsafe = cKDTree(X_unsafe)
    
    print(f"Querying {len(X_safe)} safe points within radius {d_threshold} MW...")
    # query_ball_point 返回每个 safe 点在半径 d_threshold 内的所有 unsafe 点的索引
    neighbors_list = tree_unsafe.query_ball_point(X_safe, r=d_threshold)
    
    midpoints = []
    
    for i, neighbors in enumerate(neighbors_list):
        p_safe = X_safe[i]
        for idx in neighbors:
            p_unsafe = X_unsafe[idx]
            midpoint = (p_safe + p_unsafe) / 2.0
            midpoints.append(midpoint)
            
    # 去重：由于网格的对称性，可能会有重复的或极度接近的中点
    midpoints_arr = np.array(midpoints)
    if len(midpoints_arr) > 0:
        # 保留小数点后3位进行去重，对应 1kW 的精度
        unique_midpoints = np.unique(np.round(midpoints_arr, 3), axis=0)
    else:
        unique_midpoints = midpoints_arr
        
    return unique_midpoints

def main():
    # 1. 加载网格数据
    x_path = ROOT / "data" / "case9mod_X_grid.npy"
    y_path = ROOT / "data" / "case9mod_y_grid.npy"
    
    print("Loading data...")
    X = np.load(x_path)
    y = np.load(y_path).flatten()
    
    safe_mask = (y == 1)
    unsafe_mask = (y == 0)
    
    X_safe = X[safe_mask]
    X_unsafe = X[unsafe_mask]
    
    # 2. 设定物理距离阈值 d
    # 已知网格: 180 MW / 299 steps ≈ 0.602 MW
    # 对角线距离: 0.602 * sqrt(2) ≈ 0.851 MW
    # 设定 d = 0.9 MW 完美捕获直接相邻和对角相邻的网格点
    d_threshold = 0.9 
    
    print(f"Extracting BSPs with distance threshold: {d_threshold} MW")
    bsp_midpoints = list(extract_bsp_midpoints(X_safe, X_unsafe, d_threshold))
    print(f"Extracted {len(bsp_midpoints)} unique boundary points.")
    
    # 3. 保存绝对的物理边界真值
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out_csv = results_dir / "case9mod_bsp_ground_truth.csv"
    
    df = pd.DataFrame(bsp_midpoints, columns=["P_G2_MW", "P_G3_MW"])
    df.to_csv(out_csv, index=False)
    print(f"Saved ground truth boundary to {out_csv}")
    
    # 4. 可视化验证
    figures_dir = ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)
    out_fig = figures_dir / "case9mod_bsp_ground_truth.png"
    
    plt.figure(figsize=(8, 6), dpi=150)
    # 画出安全区和不安全区作为背景 (为了清晰，进行下采样)
    plt.scatter(X_safe[::5, 0], X_safe[::5, 1], c='#e6f2ff', s=1, alpha=0.5, label='Safe Region')
    plt.scatter(X_unsafe[::5, 0], X_unsafe[::5, 1], c='#ffe6e6', s=1, alpha=0.5, label='Unsafe Region')
    
    bsp_np = np.array(bsp_midpoints)
    # 画出提取的 BSP 中点 (这是绝对的物理边界)
    plt.scatter(bsp_np[:, 0], bsp_np[:, 1], c='blue', s=3, label='BSP Midpoints (Ground Truth)')
    
    plt.title('Case9mod: Absolute Physical Boundary based on BSP')
    plt.xlabel('$P_{G2}$ (MW)')
    plt.ylabel('$P_{G3}$ (MW)')
    # 标注论文出处的参考
    plt.text(5, 175, f"Method: Topological Boundary Sample Pairs (d <= {d_threshold} MW)", 
             fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_fig)
    print(f"Saved visualization to {out_fig}")

if __name__ == "__main__":
    main()