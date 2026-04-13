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
    
    # 4. 可视化验证 (全新高对比度版本)
    figures_dir = ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)
    out_fig = figures_dir / "case9mod_bsp_ground_truth.png"
    
    plt.figure(figsize=(10, 8), dpi=300)
    
    # 画出背景色块 (安全=极淡蓝, 不安全=极淡红)，不采用下采样，全部绘制，用方块标记填满
    plt.scatter(X_unsafe[:, 0], X_unsafe[:, 1], c='#ffe6e6', s=3, marker='s', edgecolors='none', label='Unsafe Region', zorder=1)
    plt.scatter(X_safe[:, 0], X_safe[:, 1], c='#e6f2ff', s=3, marker='s', edgecolors='none', label='Safe Region', zorder=2)
    
    # 物理边界（BSP中点）画为纯黑色的线形点列，层级最高
    bsp_np = np.array(bsp_midpoints)
    plt.scatter(bsp_np[:, 0], bsp_np[:, 1], c='black', s=8, alpha=1.0, label='BSP Boundary (Ground Truth)', zorder=5)
    
    plt.title('Case9mod: Absolute Physical Boundary (Topological BSP)', fontsize=14, fontweight='bold')
    plt.xlabel('$P_{G2}$ (MW)', fontsize=12)
    plt.ylabel('$P_{G3}$ (MW)', fontsize=12)
    
    # 调整图例
    plt.legend(loc='lower right', fontsize=11, markerscale=3, framealpha=0.9, edgecolor='black')
    
    plt.xlim(0, 180)
    plt.ylim(0, 180)
    plt.grid(True, linestyle=':', alpha=0.6, color='gray')
    
    plt.tight_layout()
    plt.savefig(out_fig)
    print(f"Saved high-contrast visualization to {out_fig}")

if __name__ == "__main__":
    main()
