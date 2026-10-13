# # import re
# # import pandas as pd

# # def save_df_to_markdown(df, file_handle, title=None):
# #     """
# #     辅助函数：将DataFrame转换为Markdown表格并写入文件
# #     （手动实现，避免依赖 tabulate 库）
# #     """
# #     if title:
# #         file_handle.write(f"## {title}\n\n")
    
# #     # 1. 写入表头
# #     cols = df.columns.tolist()
# #     header = "| " + " | ".join(str(c) for c in cols) + " |"
# #     separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    
# #     file_handle.write(header + "\n")
# #     file_handle.write(separator + "\n")
    
# #     # 2. 写入数据行
# #     for _, row in df.iterrows():
# #         # 将每个单元格转为字符串，如果是浮点数保留4位小数
# #         row_str = []
# #         for val in row:
# #             if isinstance(val, float):
# #                 row_str.append(f"{val:.4f}")
# #             else:
# #                 row_str.append(str(val))
# #         line = "| " + " | ".join(row_str) + " |"
# #         file_handle.write(line + "\n")
    
# #     file_handle.write("\n") # 表格后空一行

# # def process_log_to_md(log_path, output_md_file):
# #     print(f"正在读取日志: {log_path} ...")
    
# #     # === 1. 数据解析 (核心逻辑) ===
# #     data = []
# #     current_epoch = 0
# #     strategy_counter = 0
    
# #     # 正则表达式
# #     epoch_pattern = re.compile(r'epoch \[(\d+)/\d+\]')
# #     header_pattern = re.compile(r'^\s*Class\s+I-Auroc')
# #     data_pattern = re.compile(r'^\s*([a-zA-Z0-9_]+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)')

# #     with open(log_path, 'r', encoding='utf-8') as f:
# #         for line in f:
# #             line = line.strip()
# #             # 捕获 Epoch
# #             epoch_match = epoch_pattern.search(line)
# #             if epoch_match:
# #                 current_epoch = int(epoch_match.group(1))
# #                 continue
# #             # 重置策略计数
# #             if header_pattern.match(line):
# #                 strategy_counter = 0
# #                 continue
# #             # 捕获数据
# #             data_match = data_pattern.match(line)
# #             if data_match:
# #                 strategy_counter += 1
# #                 metrics = [float(data_match.group(i)) for i in range(2, 9)]
# #                 row = {
# #                     'Class': data_match.group(1),
# #                     'Epoch': current_epoch,
# #                     'Strategy': strategy_counter,
# #                     'I-AUROC': metrics[0],
# #                     # 'I-AP': metrics[1],
# #                     # 'I-F1': metrics[2], # 如果不需要所有指标，可以注释掉以简化表格
# #                     'P-AUROC': metrics[3],
# #                     'P-AP': metrics[4],
# #                     # 'P-F1': metrics[5],
# #                     'P-AUPRO': metrics[6] # 保留一个像素级关键指标
# #                 }
# #                 data.append(row)

# #     df = pd.DataFrame(data)
# #     print(f"提取完成，共 {len(df)} 行数据。正在生成 Markdown 报告...")

# #     # === 2. 数据处理 (生成透视表) ===
    
# #     # 处理 A: 策略对比 (Strategy Comparison)
# #     # 选取最新的 Epoch (通常是最后一次测试，即 Epoch 70) 进行展示
# #     max_epoch = df['Epoch'].max()
# #     df_latest = df[df['Epoch'] == max_epoch].copy()
# #     pivot_strategy = df_latest.pivot(index='Class', columns='Strategy', values='I-AUROC')
# #     pivot_strategy.reset_index(inplace=True)
    
# #     # 处理 B: 趋势矩阵 (Trend Matrix)
# #     # 展示某个核心策略 (例如策略 5) 随时间的变化
# #     target_strat = 5
# #     df_trend = df[df['Strategy'] == target_strat].copy()
# #     pivot_trend = df_trend.pivot(index='Class', columns='Epoch', values='I-AUROC')
# #     pivot_trend.reset_index(inplace=True)

# #     # === 3. 写入 Markdown 文件 ===
# #     with open(output_md_file, 'w', encoding='utf-8') as md:
# #         md.write(f"# 异常检测实验报告\n\n")
# #         md.write(f"**数据源**: {log_path}\n")
# #         md.write(f"**总数据量**: {len(df)} 条记录\n\n")
        
# #         # 部分 1: 策略对比 (Pivot Table)
# #         md.write(f"## 1. 策略性能对比 (Epoch {max_epoch})\n")
# #         md.write("对比不同策略在最终轮次下的 I-AUROC 指标：\n\n")
# #         save_df_to_markdown(pivot_strategy, md)
        
# #         # 部分 2: 趋势分析 (Pivot Table)
# #         md.write(f"## 2. 训练趋势分析 (策略 {target_strat})\n")
# #         md.write(f"展示策略 {target_strat} 在不同训练轮数下的 I-AUROC 变化：\n\n")
# #         save_df_to_markdown(pivot_trend, md)
        
# #         # 部分 3: 全量数据矩阵 (Raw Data)
# #         # 由于数据量大，我们按 Class 和 Epoch 排序
# #         df_sorted = df.sort_values(by=['Class', 'Epoch', 'Strategy'])
# #         md.write("## 3. 全量详细数据矩阵\n")
# #         md.write("包含所有类别、轮数、策略及选定指标的完整数据：\n\n")
# #         save_df_to_markdown(df_sorted, md)

# #     print(f"报告生成成功！请查看文件: {output_md_file}")

# # # ================= 运行配置 =================
# # LOG_FILE = '/home/jjquan/experiment/vpt4/saved_results/4-32token_70epoch_seg_anomalyStyler_seghead-error_map_add/log.txt'             # 输入文件名
# # OUTPUT_MD = '/home/jjquan/experiment/vpt4/saved_results/4-32token_70epoch_seg_anomalyStyler_seghead-error_map_add/experiment_report.md' # 输出文件名

# # if __name__ == '__main__':
# #     try:
# #         process_log_to_md(LOG_FILE, OUTPUT_MD)
# #     except FileNotFoundError:
# #         print(f"错误: 找不到文件 {LOG_FILE}")











# import re
# import pandas as pd

# def parse_and_generate_avg_report(log_path, output_md_file):
#     """
#     1. 解析日志提取全量数据
#     2. 计算所有类别的平均指标 (按策略和轮数分组)
#     3. 生成格式美观的 Markdown 报告
#     """
#     print(f"正在读取日志: {log_path} ...")
    
#     # === 1. 数据解析 ===
#     data = []
#     current_epoch = 0
#     strategy_counter = 0
    
#     # 正则预编译
#     epoch_pattern = re.compile(r'epoch \[(\d+)/\d+\]')
#     header_pattern = re.compile(r'^\s*Class\s+I-Auroc')
#     data_pattern = re.compile(r'^\s*([a-zA-Z0-9_]+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)')

#     try:
#         with open(log_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 line = line.strip()
#                 # 捕获 Epoch
#                 epoch_match = epoch_pattern.search(line)
#                 if epoch_match:
#                     current_epoch = int(epoch_match.group(1))
#                     continue
#                 # 重置策略计数
#                 if header_pattern.match(line):
#                     strategy_counter = 0
#                     continue
#                 # 捕获数据行
#                 data_match = data_pattern.match(line)
#                 if data_match:
#                     strategy_counter += 1
#                     metrics = [float(data_match.group(i)) for i in range(2, 9)]
#                     row = {
#                         'Class': data_match.group(1),
#                         'Epoch': current_epoch,
#                         'Strategy': strategy_counter,
#                         'I-AUROC': metrics[0],
#                         # 'I-AP': metrics[1],
#                         # 'I-F1': metrics[2],
#                         'P-AUROC': metrics[3],
#                         'P-AP': metrics[4],
#                         # 'P-F1': metrics[5],
#                         'P-AUPRO': metrics[6]
#                     }
#                     data.append(row)
#     except FileNotFoundError:
#         print(f"错误: 找不到文件 {log_path}")
#         return

#     # 转换为 DataFrame
#     df = pd.DataFrame(data)
#     print(f"解析完成，共 {len(df)} 条原始记录。")

#     # === 2. 计算平均值 (核心逻辑) ===
#     # 按 Strategy 和 Epoch 分组，计算所有 Class 的平均值
#     avg_df = df.groupby(['Strategy', 'Epoch']).mean(numeric_only=True).reset_index()

#     # 优化显示：将 Strategy 和 Epoch 转换为整数类型（去除小数点）
#     avg_df['Strategy'] = avg_df['Strategy'].astype(int)
#     avg_df['Epoch'] = avg_df['Epoch'].astype(int)

#     # === 3. 生成 Markdown 表格 ===
#     with open(output_md_file, 'w', encoding='utf-8') as f:
#         title = "各策略在不同轮数下的平均性能指标 (所有类别均值)"
#         f.write(f"# {title}\n\n")
        
#         # 写入表头
#         cols = avg_df.columns.tolist()
#         # 创建 Markdown 表头 | Strategy | Epoch | ... |
#         header = "| " + " | ".join(str(c) for c in cols) + " |"
#         # 创建分割线 | --- | --- | ... |
#         # 为数字列设置右对齐 (---:)，为第一列设置左对齐 (---)
#         alignments = []
#         for i in range(len(cols)):
#             alignments.append("---:" if i > 0 else "---")
#         separator = "| " + " | ".join(alignments) + " |"
        
#         f.write(header + "\n")
#         f.write(separator + "\n")
        
#         # 写入数据
#         for _, row in avg_df.iterrows():
#             row_str = []
#             for col in cols:
#                 val = row[col]
#                 if isinstance(val, float):
#                     row_str.append(f"{val:.4f}") # 浮点数保留4位
#                 else:
#                     row_str.append(str(val))     # 整数直接转字符串
#             f.write("| " + " | ".join(row_str) + " |" + "\n")

#     print(f"成功生成报告: {output_md_file}")
#     # 打印前几行预览
#     print("\n结果预览:")
#     print(avg_df.head().to_string(index=False))

# # ================= 运行配置 =================
# LOG_FILE = '/home/jjquan/experiment/vpt4/saved_results/4-32token_70epoch_seg_anomalyStyler_seghead-error_map_add/log.txt'                 # 输入日志文件
# OUTPUT_MD = 'average_metrics_trend.md' # 输出 Markdown 文件

# if __name__ == '__main__':
#     parse_and_generate_avg_report(LOG_FILE, OUTPUT_MD)


    
    
import pandas as pd
import re
import os

def parse_log_file(log_path):
    """
    解析日志文件，返回包含所有原始数据的 DataFrame
    """
    print(f"正在读取日志: {log_path} ...")
    data = []
    current_epoch = 0
    strategy_counter = 0
    
    # 正则表达式定义
    epoch_pattern = re.compile(r'epoch \[(\d+)/\d+\]')
    header_pattern = re.compile(r'^\s*Class\s+I-Auroc')
    # 匹配 7 个浮点数指标
    data_pattern = re.compile(r'^\s*([a-zA-Z0-9_]+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)')

    if not os.path.exists(log_path):
        print(f"❌ 错误: 文件不存在 - {log_path}")
        return pd.DataFrame()

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # 1. 更新 Epoch
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                continue
            
            # 2. 遇到表头重置策略计数
            if header_pattern.match(line):
                strategy_counter = 0
                continue
            
            # 3. 提取数据行
            data_match = data_pattern.match(line)
            if data_match:
                strategy_counter += 1
                # 提取指标
                metrics = [float(data_match.group(i)) for i in range(2, 9)]
                
                row = {
                    'Class': data_match.group(1),
                    'Epoch': current_epoch,
                    'Strategy': strategy_counter,
                    'I-AUROC': metrics[0],
                    'I-AP':    metrics[1],
                    'I-F1':    metrics[2],
                    'P-AUROC': metrics[3],
                    'P-AP':    metrics[4],
                    'P-F1':    metrics[5],
                    'P-AUPRO': metrics[6]
                }
                data.append(row)

    df = pd.DataFrame(data)
    print(f"✅ 解析完成，共提取 {len(df)} 条记录。")
    return df

def generate_report(df, category, metrics, output_file=None):
    """
    根据指定类别和指标生成报告
    """
    if df.empty: return

    # 1. 基础列调整：Epoch 在前，Strategy 在后
    base_cols = ['Epoch', 'Strategy']
    
    # 2. 校验指标
    valid_metrics = [m for m in metrics if m in df.columns]
    if not valid_metrics:
        print(f"❌ 错误: 指定的指标 {metrics} 均不存在于数据中。")
        return

    print(f"\n📊 正在生成报告 | 类别: {category} | 指标: {valid_metrics}")

    # 3. 数据筛选与聚合
    if category == 'all':
        # 计算所有类别的平均值
        result_df = df.groupby(base_cols)[valid_metrics].mean().reset_index()
    else:
        # 筛选特定类别
        result_df = df[df['Class'] == category].copy()
        if result_df.empty:
            print(f"❌ 警告: 找不到类别 '{category}' 的数据。")
            return
        result_df = result_df[base_cols + valid_metrics]

    # 4. 数据格式化处理
    
    # 4.1 整数化
    result_df['Epoch'] = result_df['Epoch'].astype(int)
    result_df['Strategy'] = result_df['Strategy'].astype(int)

    # 4.2 百分制转换 + 强制两位小数
    for col in valid_metrics:
        result_df[col] = result_df[col].apply(lambda x: f"{x * 100:.2f}")

    # 5. 排序：先按 Epoch，再按 Strategy
    result_df = result_df.sort_values(by=['Epoch', 'Strategy'], ascending=[True, True])
    
    # 6. 生成 Markdown (设置所有列居中)
    num_cols = len(result_df.columns)
    alignments = ("center",) * num_cols
    
    md_table = result_df.to_markdown(index=False, colalign=alignments)

    # 打印预览
    print("\n--- 结果预览 (最后 5 行) ---")
    print('\n'.join(md_table.split('\n')[:2])) 
    print('\n'.join(md_table.split('\n')[-5:])) 

    # 7. 保存到文件
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# 实验报告\n\n")
            f.write(f"**类别**: {category}\n")
            f.write(f"**包含指标**: {', '.join(valid_metrics)}\n\n")
            f.write(md_table)
        print(f"\n💾 完整报告已保存至: {output_file}")

# ==========================================
# 主程序入口
# ==========================================
if __name__ == '__main__':
    # 配置
    LOG_FILE_PATH = '/home/jjquan/experiment/vpt4/mvtec_saved_results_retrain/4-32token_70epoch_seg_anomalyStyler_seghead-error_map_add/log.txt'
    OUTPUT_MD_PATH = '/home/jjquan/experiment/vpt4/mvtec_saved_results_retrain/4-32token_70epoch_seg_anomalyStyler_seghead-error_map_add/result_report.md'

    # 指定类别: 'all' 或具体类名如 'bottle'
    TARGET_CATEGORY = 'all' 

    # 指定指标
    TARGET_METRICS = ['I-AUROC', 'P-AUROC', 'P-AP','P-AUPRO']

    # 运行
    df_raw = parse_log_file(LOG_FILE_PATH)
    generate_report(df_raw, TARGET_CATEGORY, TARGET_METRICS, OUTPUT_MD_PATH)