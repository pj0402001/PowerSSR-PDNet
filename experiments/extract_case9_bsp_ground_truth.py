import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT = Path(r"D:\安全域\PowerSSR-DL")

def main():
    # 1. 从原始传统数据中加载安全点
    csv_path = Path(r"D:\安全域\1\ac_opf_9results.csv")
    print(f"Loading true physical data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    p2 = df['p2_mw'].to_numpy()
    p3 = df['p3_mw'].to_numpy()
    safe_points = np.column_stack((p2, p3))
    
    # 2. 计算网格的真实物理步长
    dp2 = (300.0 - 10.0) / 299.0
    dp3 = (270.0 - 10.0) / 299.0
    
    safe_idx_set = set()
    for i in range(len(p2)):
        idx_2 = int(round((p2[i] - 10.0) / dp2))
        idx_3 = int(round((p3[i] - 10.0) / dp3))
        safe_idx_set.add((idx_2, idx_3))
        
    # 3. 提取真实的物理边界 (BSP样本对)
    bsp_records = [] # 存储 (safe_p2, safe_p3, unsafe_p2, unsafe_p3, mid_p2, mid_p3)
    
    # 四个正交方向的邻居 (由于是密集网格，只查最近的上下左右即可保证不穿透内部)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    for (i, j) in safe_idx_set:
        curr_p2 = 10.0 + i * dp2
        curr_p3 = 10.0 + j * dp3
        
        for di, dj in directions:
            ni, nj = i + di, j + dj
            
            # 只有当邻居不在安全集合中时，它才是不安全点。
            # 【关键逻辑】：如果当前点是“内部安全点”，它的上下左右肯定都在 safe_idx_set 里，
            # 那么下面的 if 永远不会成立！所以内部点绝对不会和外部不安全点配对。
            if (ni, nj) not in safe_idx_set:
                # 只有真正的"边缘安全点"才会找到自己的"不安全邻居"
                neighbor_p2 = 10.0 + ni * dp2
                neighbor_p3 = 10.0 + nj * dp3
                
                # 构成 BSP，计算中点
                midpoint_p2 = (curr_p2 + neighbor_p2) / 2.0
                midpoint_p3 = (curr_p3 + neighbor_p3) / 2.0
                
                bsp_records.append({
                    'safe_x': curr_p2, 'safe_y': curr_p3,
                    'unsafe_x': neighbor_p2, 'unsafe_y': neighbor_p3,
                    'mid_x': midpoint_p2, 'mid_y': midpoint_p3
                })
                    
    df_bsp = pd.DataFrame(bsp_records)
    print(f"Extracted {len(df_bsp)} Boundary Sample Pairs (BSPs).")
    
    # 保存真值数据
    results_dir = ROOT / "results"
    out_csv = results_dir / "case9mod_bsp_ground_truth.csv"
    df_bsp[['mid_x', 'mid_y']].rename(columns={'mid_x':'P_G2_MW', 'mid_y':'P_G3_MW'}).drop_duplicates().to_csv(out_csv, index=False)
    
    # ==========================================
    # 4. 绘制1个全局图 + 3个局部细节图
    # ==========================================
    figures_dir = ROOT / "figures"
    out_fig = figures_dir / "case9mod_bsp_ground_truth_details.png"
    
    fig = plt.figure(figsize=(14, 12), dpi=200)
    
    # 找三个有代表性的局部区域：
    # 区域1：最顶端 (P3最大处)
    top_bsp = df_bsp.iloc[df_bsp['mid_y'].idxmax()]
    # 区域2：最右端 (P2最大处)
    right_bsp = df_bsp.iloc[df_bsp['mid_x'].idxmax()]
    # 区域3：右上角斜边界 (P2+P3最大处)
    diag_bsp = df_bsp.iloc[(df_bsp['mid_x'] + df_bsp['mid_y']).idxmax()]
    
    zoom_windows = [
        (top_bsp['mid_x'], top_bsp['mid_y'], "Detail 1: Top Edge (Max P3)"),
        (right_bsp['mid_x'], right_bsp['mid_y'], "Detail 2: Right Edge (Max P2)"),
        (diag_bsp['mid_x'], diag_bsp['mid_y'], "Detail 3: Diagonal Edge")
    ]
    window_size = 6.0 # MW 窗口大小
    
    # --- 子图 1: 全局图 ---
    ax1 = plt.subplot(2, 2, 1)
    # 画出所有的安全点
    ax1.scatter(safe_points[:, 0], safe_points[:, 1], c='#a3c2fa', s=5, label='Safe Points (AC-OPF)', edgecolors='none')
    # 画出BSP中点(黑线)
    ax1.scatter(df_bsp['mid_x'], df_bsp['mid_y'], c='black', s=2, label='BSP Midpoints (Boundary)')
    
    ax1.set_title('Global View: Exact Physical Boundary (BSP)', fontweight='bold')
    ax1.set_xlabel('$P_{G2}$ (MW)')
    ax1.set_ylabel('$P_{G3}$ (MW)')
    ax1.set_xlim(0, 190)
    ax1.set_ylim(0, 190)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower left')
    
    # 在全局图上画出三个局部框
    colors = ['red', 'green', 'purple']
    for i, (zx, zy, title) in enumerate(zoom_windows):
        rect = patches.Rectangle((zx - window_size/2, zy - window_size/2), window_size, window_size, 
                                 linewidth=2, edgecolor=colors[i], facecolor='none')
        ax1.add_patch(rect)
        ax1.text(zx, zy + window_size/2 + 2, f"Zoom {i+1}", color=colors[i], fontweight='bold', ha='center')

    # --- 子图 2,3,4: 局部细节图 ---
    for i in range(3):
        ax = plt.subplot(2, 2, i+2)
        zx, zy, title = zoom_windows[i]
        
        # 过滤出在这个窗口内的 safe_points 和 bsp
        x_min, x_max = zx - window_size/2, zx + window_size/2
        y_min, y_max = zy - window_size/2, zy + window_size/2
        
        # 画安全点
        ax.scatter(safe_points[:, 0], safe_points[:, 1], c='#a3c2fa', s=100, label='Safe Point', edgecolors='white')
        
        # 画 BSP 细节
        for _, row in df_bsp.iterrows():
            if x_min <= row['mid_x'] <= x_max and y_min <= row['mid_y'] <= y_max:
                # 1. 连线 (体现Pair关系)
                ax.plot([row['safe_x'], row['unsafe_x']], [row['safe_y'], row['unsafe_y']], 
                        'k--', linewidth=1.0, alpha=0.5)
                # 2. 画不安全点 (红色叉)
                ax.scatter(row['unsafe_x'], row['unsafe_y'], c='red', marker='x', s=80, label='Unsafe Point (Neighbor)' if _==0 else "")
                # 3. 画BSP中点 (黑色星星)
                ax.scatter(row['mid_x'], row['mid_y'], c='black', marker='*', s=150, label='BSP Midpoint' if _==0 else "", zorder=5)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(title, color=colors[i], fontweight='bold')
        ax.set_xlabel('$P_{G2}$ (MW)')
        ax.set_ylabel('$P_{G3}$ (MW)')
        ax.grid(True, linestyle='--', alpha=0.4)
        
        # 只在第二个图(第一个细节图)里加上图例，免得拥挤
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_fig)
    print(f"Saved precise details visualization to {out_fig}")

if __name__ == "__main__":
    main()
