# -*- coding: utf-8 -*-
"""
@Time    : 2025/3/21 21:37
@Author  : AadSama
@Software: Pycharm
"""
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter

def K_M(file_path):
    data = pd.read_csv(file_path)

    # 创建 Kaplan-Meier Fitter 实例
    kmf = KaplanMeierFitter()

    # 创建图形
    plt.figure(figsize=(5.5, 4))

    # 分别绘制 Score-H (group == 1) 和 Score-L (group == 0) 的生存曲线
    for group in [1, 0]:
        group_data = data[data['group'] == group]
        print(group_data)

        # 使用 Kaplan-Meier Fitter 进行生存分析
        kmf.fit(group_data['PFS'], event_observed=group_data['Status'], label=f'Score-{"H" if group == 1 else "L"}')

        # 根据组别设置颜色
        color = 'steelblue' if group == 1 else 'salmon'

        # 绘制生存曲线
        kmf.plot(ci_show=True, color=color, linewidth=1.6)


    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_linewidth(1.1)
    plt.gca().spines['left'].set_linewidth(1.1)


    # 设置图表标题和标签
    plt.title('Kaplan-Meier Survival Curve of MT-Clip', fontsize=12)


    plt.xlabel('PFS(month)')
    plt.ylabel('Probability of Survival')
    plt.legend(loc='upper right', frameon=False, fontsize=10)

    group_1_data = data[data['group'] == 1]
    group_0_data = data[data['group'] == 0]
    # 计算 Log-rank 检验的 p 值
    results = logrank_test(group_1_data['PFS'], group_0_data['PFS'], event_observed_A=group_1_data['Status'], event_observed_B=group_0_data['Status'])
    logrank_p_value = results.p_value
    logrank_p_value_formatted = f"{logrank_p_value:.4f}" if logrank_p_value >= 0.0001 else "0.0000"


    # 在右上角添加 p 值和风险比文本
    plt.text(25, 0.65, f"p-value = {logrank_p_value_formatted}", ha='right', fontsize=12, color='black')

    # 计算风险比 (HR) 和置信区间
    cph = CoxPHFitter()
    data['group'] = data['group'].astype('category')  # 确保 group 列是类别型
    cph.fit(data[['PFS', 'Status', 'group']], duration_col='PFS', event_col='Status')
    hr = cph.hazard_ratios_['group']
    hr_formatted = f"{1/hr:.1f}"


    plt.text(25, 0.55, f"HR = {hr_formatted}", ha='right', fontsize=12, color='black')

    print(f"HR = {hr_formatted}")
    print(f"p = {logrank_p_value:.4f}")

    plt.savefig("results/K_M.svg", dpi=300)
    plt.show()