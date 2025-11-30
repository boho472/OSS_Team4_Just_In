import json
import os

# ================================
# 기본 파일 초기화/로드
# ================================
def reset_tracker_log(save_json_path):
    with open(save_json_path, "w") as f:
        f.write("{}")   # 비어 있는 JSON dict로 초기화


def load_tracker_log(save_json_path):
    if not os.path.exists(save_json_path):
        reset_tracker_log(save_json_path)
        return {}

    with open(save_json_path, "r") as f:
        try:
            return json.load(f)
        except:
            reset_tracker_log(save_json_path)
            return {}


def save_tracker_log(save_json_path,log):
    with open(save_json_path, "w") as f:
        json.dump(log, f, indent=2)


# ================================
# 원하는 구조로 프레임 기록 (누적 저장)
# ================================
def update_tracker_log(save_json_path,frame_num,result_dict):
    """
    result_dict 구조:
    {
      "frame_000038": {
        "tracks_1": {
          "created_frame": ,
          "last_detected_frame": ,
          "undetected_num": ,
          "det_bbox": {"x": ,"y": ,"w": ,"h": }
          "status": 
        },
        "tracks_2": {
          "created_frame": 38,
          "last_detected_frame": 38,
          "undetected_num": 0,
          "det_bbox": x, y, w, h
          "status": "detected"
        },
        "dead": []
      }
    }
    """
    
    # 기존 전체 로그 불러오기
    log = load_tracker_log(save_json_path)

    # 이번 프레임 데이터 초기화

    # -----------------------------
    # 3. 기존 로그에 프레임만 추가 (누적!)
    # -----------------------------
    log[frame_num] = result_dict[frame_num]

    # 저장
    save_tracker_log(save_json_path,log)