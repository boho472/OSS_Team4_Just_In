import numpy as np
from json_system.tracker_log import update_tracker_log

def ht_result(tracker,dataset,i,saved_frame,new_info,new_info_dict,dict_key,save_json_log_path):
    new_obj = tracker.new_obj
    
    with open(dataset.ob_txt_path) as f:
        obj_info = f.readlines()
        if not obj_info:  # ✅ 빈 파일 체크
            print(f"⚠️ Warning: Empty TXT file at frame {saved_frame[i]}")
            whole_txt_file = np.array([]).reshape(0, 16)
        else:
            whole_txt_file = np.array([item.strip().split(' ') for item in obj_info])
            
            # ✅ 1차원 배열을 2차원으로 변환
            if whole_txt_file.ndim == 1:
                whole_txt_file = whole_txt_file.reshape(1, -1)
    
    print(f"📊 [Frame {saved_frame[i]}] TXT shape: {whole_txt_file.shape}, new_obj: {len(new_obj)}")
    
    for (obj_num,obj_id) in new_obj:
        obj_id_str = "tracks_" + str(obj_id)
        
        print(f"   Processing obj_num={obj_num}, obj_id={obj_id}")
        
        # ✅ 안전한 삭제
        if obj_id in tracker.current_frame_ids:
            tracker.current_frame_ids.remove(obj_id)
        else:
            print(f"   ⚠️ Warning: obj_id={obj_id} not in current_frame_ids")
        
        if obj_id in tracker.previous_frame_ids:
            tracker.previous_frame_ids.remove(obj_id)
        
        # ✅ obj_num 범위 체크
        if obj_num >= whole_txt_file.shape[0]:
            print(f"   ❌ Error: obj_num={obj_num} >= txt_lines={whole_txt_file.shape[0]}")
            continue
        
        # ✅ 컬럼 수 체크
        if whole_txt_file.shape[1] < 8:
            print(f"   ❌ Error: Not enough columns in TXT (expected 8+, got {whole_txt_file.shape[1]})")
            continue
        
        for key, value in new_info_dict.items():
            if obj_id_str == key:
                value["last_detected_frame"] = int(saved_frame[i][:-4])
                value["undetected_num"] = 0
                value["det_bbox"]["x"] = float(whole_txt_file[obj_num, 4])
                value["det_bbox"]["y"] = float(whole_txt_file[obj_num, 5])
                value["det_bbox"]["w"] = float(whole_txt_file[obj_num, 6])
                value["det_bbox"]["h"] = float(whole_txt_file[obj_num, 7])
                value["status"] = "detected"
        
        if obj_id_str not in dict_key:
            new_dict = {}
            new_dict[obj_id_str] = {}
            new_dict[obj_id_str]["created_frame"] = int(saved_frame[i][:-4])
            new_dict[obj_id_str]["last_detected_frame"] = int(saved_frame[i][:-4])
            new_dict[obj_id_str]["undetected_num"] = 0
            new_dict[obj_id_str]["det_bbox"] = {}
            new_dict[obj_id_str]["det_bbox"]["x"] = float(whole_txt_file[obj_num, 4])
            new_dict[obj_id_str]["det_bbox"]["y"] = float(whole_txt_file[obj_num, 5])
            new_dict[obj_id_str]["det_bbox"]["w"] = float(whole_txt_file[obj_num, 6])
            new_dict[obj_id_str]["det_bbox"]["h"] = float(whole_txt_file[obj_num, 7])
            new_dict[obj_id_str]["status"] = "detected"
            new_info.append(new_dict)
    
    for j in tracker.previous_frame_ids:
        id_str = "tracks_" + str(j)
        for info in new_info:
            if id_str in info:
                info[id_str]["undetected_num"] += 1
                info[id_str]["det_bbox"]["x"] = 0.0
                info[id_str]["det_bbox"]["y"] = 0.0
                info[id_str]["det_bbox"]["w"] = 0.0
                info[id_str]["det_bbox"]["h"] = 0.0
                info[id_str]["status"] = "undetected"
    
    new_info_dict = {}
    for item_dict in new_info:
        for key, value in item_dict.items():
            new_info_dict[key] = value
    dict_key = list(new_info_dict.keys())
    
    new_info_dict["dead"] = tracker.dead_list
    result_dict = {}
    frame_num = "frame_" + saved_frame[i][:-4]
    result_dict[frame_num] = new_info_dict

    #=======================
    # tracker_log 업데이트(HT 전체 로그)
    #======================= 
    update_tracker_log(save_json_log_path,frame_num,result_dict)
    
    print(result_dict, "\n")
    
    return new_info,new_info_dict,dict_key