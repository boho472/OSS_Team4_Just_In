import numpy as np
import os
import json
import matplotlib.cm as cm
import matplotlib.colors as colors

# PLY 파일 헤더 정의
PLY_HEADER = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""

def generate_random_color(obj_id):
    """
    객체 ID를 기반으로 고유한 색상(RGB)을 생성합니다.
    (0~255 범위의 정수 튜플)
    """
    # 객체 ID를 시드로 사용하여 일관된 무작위 색상 생성
    np.random.seed(obj_id)
    color = np.random.randint(0, 255, 3)
    return tuple(color.tolist())

def get_color_map(obj_ids):
    """
    주어진 객체 ID 리스트에 대해 색상을 할당하는 딕셔너리를 생성합니다.
    """
    # 0번 ID는 일반적으로 배경 또는 매칭되지 않은 객체에 사용될 수 있으므로
    # 1번부터 시작하도록 seed를 조정하거나, 단순히 ID 자체를 사용합니다.
    
    # 트래킹 ID에 따라 고유한 색상을 할당합니다.
    color_map = {}
    for obj_id in sorted(list(set(obj_ids))):
        if obj_id is not None:
            # matplotlib의 컬러 맵을 사용하거나, 고정된 무작위 색상 사용
            # 여기서는 객체 ID 기반의 무작위 색상을 사용합니다.
            color_map[obj_id] = generate_random_color(obj_id)
            
    return color_map

def convert_3d_to_ply(json_path, ply_save_path, frame_idx, color_map):
    """
    JSON 파일에서 3D 좌표 및 객체 정보를 읽어 PLY 파일로 변환하여 저장합니다.

    Args:
        json_path (str): 프레임별 YOLO, 3D 변환 정보가 담긴 JSON 파일 경로.
        ply_save_path (str): PLY 파일을 저장할 디렉토리 경로.
        frame_idx (int): 현재 프레임 번호.
        color_map (dict): {obj_id: (R, G, B)} 형태의 색상 맵.
    """
    if not os.path.exists(json_path):
        print(f"⚠️ JSON 파일이 존재하지 않습니다: {json_path}")
        return

    with open(json_path, 'r') as f:
        frame_data = json.load(f)

    # model_3D_convert.py의 normalization_3d 키에서 3D 좌표를 가져옵니다.
    norm3d_dict = frame_data.get("normalization_3d", {})
    
    # model_DAM4SAM.py의 dam4sam 키에서 HT object ID를 가져옵니다.
    # HybridTrack/DAM4SAM이 통합된 경우에만 이 정보를 사용합니다.
    dam4sam_data = frame_data.get("dam4sam", {})
    dam_masks = dam4sam_data.get("masks", [])
    
    # 3D 좌표 데이터 준비
    points_data = []
    
    for det_id, norm3d_info in norm3d_dict.items():
        X = norm3d_info.get('x')
        Y = norm3d_info.get('y')
        Z = norm3d_info.get('z')

        if X is None or Y is None or Z is None:
            continue
        
        # 기본 색상: 회색
        R, G, B = 100, 100, 100
        
        # DAM4SAM 데이터가 있다면 HT object ID를 찾아 색상 적용
        for mask_info in dam_masks:
            # det_id와 매칭되는 정보를 찾기 위한 복잡한 로직이 필요할 수 있습니다.
            # model_DAM4SAM.py에서는 HT-ID만 관리하고 있으므로, 여기서는 임시적으로
            # det_id의 인덱스(det_0 -> 0)와 dam_masks의 인덱스를 일치시킨다고 가정합니다.
            
            # 실제로는 det_id와 ht_object_id를 연결하는 정보가 필요하지만,
            # 현재 코드 구조상 직접적인 매핑이 없으므로, norm3d_dict의 순서를 따라
            # dam_masks의 ht_object_id를 할당합니다. (불완전한 가정)
            
            # 가장 확실한 방법: json_path에 저장된 DAM4SAM_results와 normalization_3d를
            # internal_id 또는 det_id를 통해 명시적으로 연결하는 정보가 필요합니다.
            
            # 임시 방편으로, det_id의 인덱스를 사용하여 매칭을 시도합니다.
            # det_0, det_1, ...
            try:
                # 'det_1' -> 1
                det_idx = int(det_id.split('_')[1])
            except:
                det_idx = -1
                
            # DAM4SAM 결과 리스트에서 해당 인덱스의 객체를 찾습니다.
            if det_idx >= 0 and det_idx < len(dam_masks):
                # dam_masks의 내부 ID는 DAM4SAM의 internal_id를 사용합니다.
                # 그러나 PLY 시각화에서는 HybridTrack의 object_id(ht_object_id)를
                # 기준으로 색상을 부여하는 것이 추적 결과 시각화에 더 적합합니다.
                ht_obj_id = dam_masks[det_idx].get('ht_object_id')
                if ht_obj_id is not None and ht_obj_id in color_map:
                    R, G, B = color_map[ht_obj_id]
                    break
        
        # (X, Y, Z, R, G, B) 형태로 저장
        points_data.append(f"{X:.4f} {Y:.4f} {Z:.4f} {R} {G} {B}")

    num_points = len(points_data)
    
    # PLY 파일 내용 생성
    ply_content = PLY_HEADER.format(num_points)
    ply_content += "\n".join(points_data)
    
    # 파일 저장
    file_name = f"{frame_idx:06d}.ply"
    save_path = os.path.join(ply_save_path, file_name)
    
    with open(save_path, "w") as f:
        f.write(ply_content)

    print(f"💾 3D Point Cloud saved to: {save_path} ({num_points} points)")


if __name__ == '__main__':
    # 이 부분은 독립적인 테스트용 예시이며, 실제 추적 파이프라인에서는
    # tracking_main.py에서 호출됩니다.
    
    # 임시 폴더 구조 설정 (실제 환경에 맞게 수정 필요)
    # base_dir = "./results/0000"
    # save_json_path = os.path.join(base_dir, "json")
    # point_cloud_frame = os.path.join(base_dir, "point_cloud_frame")
    
    # os.makedirs(point_cloud_frame, exist_ok=True)
    
    # # 임시 JSON 파일 경로 (예시)
    # frame_num = 38
    # json_file = f"{frame_num:06d}.json"
    # json_path = os.path.join(save_json_path, json_file)
    
    # # 임시 객체 ID 및 색상 맵 (실제로는 HT 로그에서 추출되어야 함)
    # sample_color_map = get_color_map([1, 2, 3])
    
    # print(f"Sample Color Map: {sample_color_map}")

    # # 함수 호출 (JSON 파일이 실제로 존재해야 함)
    # # convert_3d_to_ply(json_path, point_cloud_frame, frame_num, sample_color_map)
    
    print("This file is intended to be imported and called by tracking_main.py.")