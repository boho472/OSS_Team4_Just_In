import motmetrics as mm
import numpy as np
import os
import pandas as pd

def evaluate_mot(gt_path, result_path, seq_name):
    """MOT 평가"""
    print(f"\n{'='*60}")
    print(f"평가: {seq_name}")
    print(f"{'='*60}")
    print(f"GT: {gt_path}")
    print(f"Result: {result_path}")
    
    # 파일 존재 확인
    if not os.path.exists(gt_path):
        print(f"❌ GT 파일 없음: {gt_path}")
        return None
    if not os.path.exists(result_path):
        print(f"❌ Result 파일 없음: {result_path}")
        return None
    
    # 데이터 로드
    try:
        gt = mm.io.loadtxt(gt_path, fmt='mot15-2D')
        hyp = mm.io.loadtxt(result_path, fmt='mot15-2D')
    except Exception as e:
        print(f"❌ 파일 로드 실패: {e}")
        return None
    
    print(f"✅ GT frames: {len(gt.index.unique())}, objects: {len(gt)}")
    print(f"✅ Result frames: {len(hyp.index.unique())}, objects: {len(hyp)}")
    
    # Accumulator 생성
    acc = mm.MOTAccumulator(auto_id=True)
    
    # 모든 프레임
    all_frames = sorted(set(gt.index.unique()) | set(hyp.index.unique()))
    
    # 프레임별 비교
    for frame in all_frames:
        # GT
        if frame in gt.index:
            gt_data = gt.loc[frame]
            if isinstance(gt_data, pd.Series):
                gt_data = gt_data.to_frame().T
            gt_ids = gt_data.index.values
            gt_boxes = gt_data[['X', 'Y', 'Width', 'Height']].values
        else:
            gt_ids = np.array([])
            gt_boxes = np.empty((0, 4))
        
        # Hypothesis
        if frame in hyp.index:
            hyp_data = hyp.loc[frame]
            if isinstance(hyp_data, pd.Series):
                hyp_data = hyp_data.to_frame().T
            hyp_ids = hyp_data.index.values
            hyp_boxes = hyp_data[['X', 'Y', 'Width', 'Height']].values
        else:
            hyp_ids = np.array([])
            hyp_boxes = np.empty((0, 4))
        
        # IoU 계산
        distances = mm.distances.iou_matrix(gt_boxes, hyp_boxes, max_iou=0.5)
        acc.update(gt_ids, hyp_ids, distances)
    
    # 지표 계산
    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=['num_switches', 'mota', 'motp', 'idf1', 'precision', 'recall'],
        name=seq_name
    )
    
    return summary

# 평가 실행
base_path = '/content/drive/MyDrive/Multi_Tracking_Person/src/data'

gt_path = f'{base_path}/ground_truth/0000/gt.txt'
result_path = f'{base_path}/pipeline_mot_result/0000.txt'

summary = evaluate_mot(gt_path, result_path, 'PETS09-S2L2')

if summary is not None:
    print("\n" + "="*60)
    print("📊 평가 결과")
    print("="*60)
    print(summary)
    print("\n🎯 핵심 지표:")
    print(f"  - ID Switches (IDs): {summary['num_switches'].values[0]:.0f}")
    print(f"  - MOTA: {summary['mota'].values[0]*100:.1f}%")
    print(f"  - IDF1: {summary['idf1'].values[0]*100:.1f}%")