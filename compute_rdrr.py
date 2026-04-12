"""计算相对退化减少率（RDRR）并生成汇总报告。

读取 robust_results/ 下所有 JSON 结果文件，以 clean.json 为基准，
计算各扰动条件下的退化幅度和 RDRR 指标，输出 Markdown 格式汇总表格。

用法:
    python compute_rdrr.py --results-dir robust_results
    python compute_rdrr.py --results-dir robust_results --baseline-dir baseline_results
"""

import os
import json
import argparse
from collections import defaultdict


def load_results(results_dir):
    """加载所有 JSON 结果文件。"""
    results = {}
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith('.json'):
            continue
        fpath = os.path.join(results_dir, fname)
        with open(fpath, 'r') as f:
            data = json.load(f)
        key = fname.replace('.json', '')
        results[key] = data
    return results


def compute_degradation(clean_metrics, noisy_metrics, metric_name):
    """计算退化幅度: Degradation = P(clean) - P(noisy)"""
    return clean_metrics[metric_name] - noisy_metrics[metric_name]


def compute_rdrr(deg_baseline, deg_ours):
    """计算 RDRR = (Deg_baseline - Deg_ours) / Deg_baseline × 100%"""
    if abs(deg_baseline) < 1e-8:
        return 0.0
    return (deg_baseline - deg_ours) / deg_baseline * 100.0


def format_table_row(cells, widths=None):
    """格式化表格行。"""
    if widths:
        formatted = [str(c).ljust(w) for c, w in zip(cells, widths)]
    else:
        formatted = [str(c) for c in cells]
    return '| ' + ' | '.join(formatted) + ' |'


def main():
    parser = argparse.ArgumentParser(description='Compute RDRR metrics')
    parser.add_argument('--results-dir', required=True, help='Directory containing result JSON files')
    parser.add_argument('--baseline-dir', default='', help='Optional: directory with baseline results for RDRR')
    args = parser.parse_args()

    results = load_results(args.results_dir)

    if 'clean' not in results:
        print('[ERROR] clean.json not found in results directory!')
        print('Available results:', list(results.keys()))
        return

    clean = results['clean']['metrics']

    # 如果提供了基线目录，加载基线结果
    baseline_results = None
    baseline_clean = None
    if args.baseline_dir and os.path.exists(args.baseline_dir):
        baseline_results = load_results(args.baseline_dir)
        if 'clean' in baseline_results:
            baseline_clean = baseline_results['clean']['metrics']

    print('=' * 80)
    print('SparseBEV 鲁棒性基准测试结果汇总')
    print('=' * 80)

    excel_data = []

    def add_to_excel(category, experiment, param, metrics, baseline_rdrr=None):
        row = {
            'Category': category,
            'Experiment': experiment,
            'Parameter': param,
            'NDS': metrics.get('NDS', 0),
            'mAP': metrics.get('mAP', 0),
            'mATE': metrics.get('mATE', 0),
            'mASE': metrics.get('mASE', 0),
            'mAOE': metrics.get('mAOE', 0),
            'mAVE': metrics.get('mAVE', 0),
            'mAAE': metrics.get('mAAE', 0),
            'NDS_Degradation': compute_degradation(clean, metrics, 'NDS') if category != 'Clean' else 0.0,
            'mAP_Degradation': compute_degradation(clean, metrics, 'mAP') if category != 'Clean' else 0.0,
        }
        if baseline_rdrr is not None:
            row.update(baseline_rdrr)
        excel_data.append(row)

    # ---- 基准结果 ----
    add_to_excel('Clean', 'Clean Baseline', '-', clean)
    print('\n## 基准结果 (Clean)')
    print(f'  NDS:  {clean["NDS"]:.4f}')
    print(f'  mAP:  {clean["mAP"]:.4f}')
    print(f'  mATE: {clean["mATE"]:.4f}')
    print(f'  mASE: {clean["mASE"]:.4f}')
    print(f'  mAOE: {clean["mAOE"]:.4f}')
    print(f'  mAVE: {clean["mAVE"]:.4f}')
    print(f'  mAAE: {clean["mAAE"]:.4f}')

    # ---- 丢帧测试结果 ----
    drop_results = {k: v for k, v in results.items() if k.startswith('drop_')}
    if drop_results:
        print('\n## 丢帧测试结果')
        print()
        header = ['Drop Rate', 'NDS', 'mAP', 'NDS_Deg', 'mAP_Deg']
        if baseline_results:
            header.extend(['NDS_RDRR(%)', 'mAP_RDRR(%)'])
        print(format_table_row(header))
        print('|' + '|'.join(['-' * (len(h) + 2) for h in header]) + '|')

        for key in sorted(drop_results.keys(), key=lambda x: int(x.split('_')[1])):
            m = drop_results[key]['metrics']
            ratio = drop_results[key].get('drop_ratio', key.split('_')[1])
            nds_deg = compute_degradation(clean, m, 'NDS')
            map_deg = compute_degradation(clean, m, 'mAP')

            row = [f'{ratio}%', f'{m["NDS"]:.4f}', f'{m["mAP"]:.4f}',
                   f'{nds_deg:.4f}', f'{map_deg:.4f}']

            if baseline_results and key in baseline_results:
                bm = baseline_results[key]['metrics']
                b_nds_deg = compute_degradation(baseline_clean, bm, 'NDS')
                b_map_deg = compute_degradation(baseline_clean, bm, 'mAP')
                nds_rdrr = compute_rdrr(b_nds_deg, nds_deg)
                map_rdrr = compute_rdrr(b_map_deg, map_deg)
                row.extend([f'{nds_rdrr:.2f}', f'{map_rdrr:.2f}'])
                rdrr_dict = {'NDS_RDRR(%)': nds_rdrr, 'mAP_RDRR(%)': map_rdrr}
            else:
                rdrr_dict = None

            add_to_excel('Drop', f'Drop {ratio}%', f'{ratio}%', m, rdrr_dict)
            print(format_table_row(row))

    # ---- 外参扰动测试结果 ----
    ext_single = {k: v for k, v in results.items() 
                  if k.startswith('extrinsics_') and k.endswith('_single')}
    ext_all = {k: v for k, v in results.items() 
               if k.startswith('extrinsics_') and k.endswith('_all')}

    for label, ext_results in [('Single Camera', ext_single), ('All Cameras', ext_all)]:
        if ext_results:
            print(f'\n## 外参扰动测试结果 ({label})')
            print()
            header = ['Level', 'NDS', 'mAP', 'NDS_Deg', 'mAP_Deg']
            if baseline_results:
                header.extend(['NDS_RDRR(%)', 'mAP_RDRR(%)'])
            print(format_table_row(header))
            print('|' + '|'.join(['-' * (len(h) + 2) for h in header]) + '|')

            for key in sorted(ext_results.keys()):
                m = ext_results[key]['metrics']
                level = ext_results[key].get('extrinsics_level', key.split('_')[1])
                nds_deg = compute_degradation(clean, m, 'NDS')
                map_deg = compute_degradation(clean, m, 'mAP')

                row = [level, f'{m["NDS"]:.4f}', f'{m["mAP"]:.4f}',
                       f'{nds_deg:.4f}', f'{map_deg:.4f}']

                if baseline_results and key in baseline_results:
                    bm = baseline_results[key]['metrics']
                    b_nds_deg = compute_degradation(baseline_clean, bm, 'NDS')
                    b_map_deg = compute_degradation(baseline_clean, bm, 'mAP')
                    nds_rdrr = compute_rdrr(b_nds_deg, nds_deg)
                    map_rdrr = compute_rdrr(b_map_deg, map_deg)
                    row.extend([f'{nds_rdrr:.2f}', f'{map_rdrr:.2f}'])
                    rdrr_dict = {'NDS_RDRR(%)': nds_rdrr, 'mAP_RDRR(%)': map_rdrr}
                else:
                    rdrr_dict = None

                add_to_excel('Extrinsics', f'{label} {level}', f'{level}', m, rdrr_dict)
                print(format_table_row(row))

    # ---- 遮挡测试结果 ----
    occ_results = {k: v for k, v in results.items() if k.startswith('occlusion_')}
    if occ_results:
        print('\n## 遮挡测试结果')
        print()
        exp_to_level = {1.0: 'S1', 2.0: 'S2', 3.0: 'S3', 5.0: 'S4'}
        header = ['Level', 'Exp', 'NDS', 'mAP', 'NDS_Deg', 'mAP_Deg']
        if baseline_results:
            header.extend(['NDS_RDRR(%)', 'mAP_RDRR(%)'])
        print(format_table_row(header))
        print('|' + '|'.join(['-' * (len(h) + 2) for h in header]) + '|')

        for key in sorted(occ_results.keys(), 
                         key=lambda x: float(x.split('exp')[1])):
            m = occ_results[key]['metrics']
            exp_val = occ_results[key].get('occlusion_exp', 
                                            float(key.split('exp')[1]))
            level = exp_to_level.get(exp_val, f'S?')
            nds_deg = compute_degradation(clean, m, 'NDS')
            map_deg = compute_degradation(clean, m, 'mAP')

            row = [level, f'{exp_val}', f'{m["NDS"]:.4f}', f'{m["mAP"]:.4f}',
                   f'{nds_deg:.4f}', f'{map_deg:.4f}']

            if baseline_results and key in baseline_results:
                bm = baseline_results[key]['metrics']
                b_nds_deg = compute_degradation(baseline_clean, bm, 'NDS')
                b_map_deg = compute_degradation(baseline_clean, bm, 'mAP')
                nds_rdrr = compute_rdrr(b_nds_deg, nds_deg)
                map_rdrr = compute_rdrr(b_map_deg, map_deg)
                row.extend([f'{nds_rdrr:.2f}', f'{map_rdrr:.2f}'])
                rdrr_dict = {'NDS_RDRR(%)': nds_rdrr, 'mAP_RDRR(%)': map_rdrr}
            else:
                rdrr_dict = None

            add_to_excel('Occlusion', f'Occlusion {level}', f'exp={exp_val}', m, rdrr_dict)
            print(format_table_row(row))

    # ---- 保存汇总到文件 ----
    summary = {
        'clean': clean,
        'tests': {}
    }

    for key, data in results.items():
        if key == 'clean':
            continue
        m = data['metrics']
        entry = {
            'NDS': m['NDS'],
            'mAP': m['mAP'],
            'NDS_degradation': compute_degradation(clean, m, 'NDS'),
            'mAP_degradation': compute_degradation(clean, m, 'mAP'),
        }
        summary['tests'][key] = entry

    summary_path = os.path.join(args.results_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n汇总 JSON 结果已保存到: {summary_path}')

    # ---- 生成 Excel 报告 ----
    try:
        import pandas as pd
        df = pd.DataFrame(excel_data)
        excel_path = os.path.join(args.results_dir, 'summary.xlsx')
        try:
            df.to_excel(excel_path, index=False)
            print(f'Excel 汇总文件已保存到: {excel_path}')
        except ImportError:
            # 降级生成 CSV
            csv_path = os.path.join(args.results_dir, 'summary.csv')
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f'\n[提示] 当前环境缺少 openpyxl 库无法生成 .xlsx，已降级生成 CSV 文件: {csv_path}')
            print('您可以直接使用 Excel 打开此 CSV 文件。')
    except ImportError:
        print('\n[提示] 环境中未安装 pandas 库，跳过生成 Excel 文件。')


if __name__ == '__main__':
    main()
