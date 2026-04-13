import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r"D:\安全域\PowerSSR-DL")

def main():
    # 1. 从最原始的传统数据中加载安全点
    csv_path = Path(r"D:\安全域\1\ac_opf_9results.csv")
    print(f"Loading true physical data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 提取有效的坐标 (P_G2, P_G3)
    p2 = df['p2_mw'].to_numpy()
    p3 = df['p3_mw'].to_numpy()
    safe_points = np.column_stack((p2, p3))
    
    # 2. 计算网格的真实物理步长
    # 原始扫描配置为 p2: 10~300 (300 steps), p3: 10~270 (300 steps)
    dp2 = (300.0 - 10.0) / 299.0
    dp3 = (270.0 - 10.0) / 299.0
    print(f"Original physical grid step: dp2 = {dp2:.4f} MW, dp3 = {dp3:.4f} MW")
    
    # 将安全点放入一个以整数索引的集合中
    safe_idx_set = set()
    for i in range(len(p2)):
        idx_2 = int(round((p2[i] - 10.0) / dp2))
        idx_3 = int(round((p3[i] - 10.0) / dp3))
        safe_idx_set.add((idx_2, idx_3))
        
    print(f"Total safe points mapped to grid: {len(safe_idx_set)}")
    
    # 3. 提取真实的物理边界 (BSP中点)
    bsp_midpoints = []
    
    # 四个正交方向的邻居
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    for (i, j) in safe_idx_set:
        # 还原当前安全点的真实坐标
        curr_p2 = 10.0 + i * dp2
        curr_p3 = 10.0 + j * dp3
        
        for di, dj in directions:
            ni, nj = i + di, j + dj
            
            # 只要邻居不在安全集合中，它就是一个"不安全点" (构成了BSP边界)
            if (ni, nj) not in safe_idx_set:
                neighbor_p2 = 10.0 + ni * dp2
                neighbor_p3 = 10.0 + nj * dp3
                
                # 构成 BSP，计算中点
                midpoint_p2 = (curr_p2 + neighbor_p2) / 2.0
                midpoint_p3 = (curr_p3 + neighbor_p3) / 2.0
                bsp_midpoints.append([midpoint_p2, midpoint_p3])
                    
    bsp_np = np.array(bsp_midpoints)
    print(f"Extracted {len(bsp_np)} absolute physical boundary points (BSP midpoints).")
    
    # 4. 保存真值数据
    results_dir = ROOT / "results"
    out_csv = results_dir / "case9mod_bsp_ground_truth.csv"
    df_out = pd.DataFrame(bsp_np, columns=["P_G2_MW", "P_G3_MW"])
    df_out.to_csv(out_csv, index=False)
    print(f"Saved exact ground truth to {out_csv}")
    
    # 5. 高清可视化 (纯净的 AC-OPF 物理边界)
    figures_dir = ROOT / "figures"
    out_fig = figures_dir / "case9mod_bsp_ground_truth.png"
    
    plt.figure(figsize=(10, 8), dpi=300)
    
    # 内部的安全点 (浅蓝色)
    plt.scatter(safe_points[:, 0], safe_points[:, 1], c='#a3c2fa', s=20, label='Feasible Region Inside (AC-OPF)', edgecolors='none', zorder=2)
    
    # 边界的 BSP中点 (黑色轮廓)
    plt.scatter(bsp_np[:, 0], bsp_np[:, 1], c='black', s=12, alpha=1.0, label='Exact Physical Boundary (BSP midpoints)', zorder=5)
    
    plt.title('Case9mod: Exact Physical Boundary from Traditional AC-OPF (BSP)', fontsize=14, fontweight='bold')
    plt.xlabel('$P_{G2}$ (MW)', fontsize=12)
    plt.ylabel('$P_{G3}$ (MW)', fontsize=12)
    
    # 标注出明确的物理红线 (发电量下限 10 MW)
    plt.axvline(x=10, color='red', linestyle='--', linewidth=1.5, zorder=6, label='Generator Min Limit (10 MW)')
    plt.axhline(y=10, color='red', linestyle='--', linewidth=1.5, zorder=6)
    
    plt.xlim(0, 200)  # 只框定有数据的部分
    plt.ylim(0, 200)
    plt.grid(True, linestyle=':', alpha=0.6, color='gray', zorder=1)
    
    plt.legend(loc='upper right', fontsize=11, markerscale=1.5, framealpha=0.9, edgecolor='black')
    
    plt.tight_layout()
    plt.savefig(out_fig)
    print(f"Saved precise visualization to {out_fig}")

if __name__ == "__main__":
    main()
