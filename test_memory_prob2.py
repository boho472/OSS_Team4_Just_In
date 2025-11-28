from tracking_wrapper_mot import DAM4SAMMOT
import torch
import numpy as np
import cv2
from PIL import Image
import psutil
import GPUtil
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import sys


class MemoryMonitor:
    def __init__(self):
        self.cpu_memory = []
        self.gpu_memory = []
        self.timestamps = []
        self.num_objects = []
        self.start_time = time.time()

    def record(self, num_objs):
        # CPU 메모리
        process = psutil.Process()
        cpu_mem_mb = process.memory_info().rss / 1024 / 1024

        # GPU 메모리
        try:
            gpus = GPUtil.getGPUs()
            gpu_mem_mb = gpus[0].memoryUsed if gpus else 0
        except:
            gpu_mem_mb = torch.cuda.memory_allocated() / 1024 / \
                1024 if torch.cuda.is_available() else 0

        elapsed = time.time() - self.start_time

        self.cpu_memory.append(cpu_mem_mb)
        self.gpu_memory.append(gpu_mem_mb)
        self.timestamps.append(elapsed)
        self.num_objects.append(num_objs)

        return cpu_mem_mb, gpu_mem_mb

    def plot(self, title="Memory Usage"):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # 메모리 사용량
        ax1.plot(self.timestamps, self.cpu_memory,
                 label='CPU Memory (MB)', marker='o')
        ax1.plot(self.timestamps, self.gpu_memory,
                 label='GPU Memory (MB)', marker='s')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Memory (MB)')
        ax1.set_title(f'{title} - Memory Usage')
        ax1.legend()
        ax1.grid(True)

        # 객체 수
        ax2.plot(self.timestamps, self.num_objects,
                 label='Number of Objects', marker='x', color='red')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Number of Objects')
        ax2.set_title(f'{title} - Object Count')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(f'{title.replace(" ", "_")}.png', dpi=150)
        plt.show()

# 4. 가짜 이미지 생성 (실험용)


def generate_dummy_image(width=1280, height=720):
    """테스트용 더미 이미지 생성"""
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(img)


def generate_moving_bbox(frame_idx, obj_id, total_frames=1000):
    """화면을 가로지르는 bbox 생성 (화면 밖으로 나가게)"""
    progress = frame_idx / total_frames
    x = int(progress * 1280) - 100  # 화면 왼쪽에서 오른쪽으로
    y = 200 + (obj_id * 50) % 400
    w, h = 100, 150

    # 화면 밖으로 나가면 더 이상 보이지 않음
    if x > 1280 or x < -100:
        return None

    return [max(0, x), y, w, h]


def experiment_with_active_dead_ids():
    """
    실험 C: active_track_ids와 dead_track_ids 파라미터 사용
    """
    print("\n" + "=" * 60)
    print("실험 C: active/dead IDs 파라미터 사용 (메모리 관리)")
    print("=" * 60)

    tracker = DAM4SAMMOT(model_size='tiny', checkpoint_dir='./checkpoints')
    monitor = MemoryMonitor()

    image = generate_dummy_image()
    init_regions = [
        {'bbox': [100, 100, 100, 150]},
        {'bbox': [300, 200, 100, 150]}
    ]
    tracker.initialize(image, init_regions)

    total_frames = 500
    add_object_every = 10
    remove_after_frames = 50

    object_lifetime = {}  # {obj_id: birth_frame}

    print(f"\n총 프레임: {total_frames}")
    print(f"새 객체 추가 주기: {add_object_every} 프레임")
    print(f"객체 수명: {remove_after_frames} 프레임")
    print(f"예상 안정 객체 수: ~{remove_after_frames // add_object_every + 2}개\n")

    for frame_idx in range(1, total_frames + 1):
        image = generate_dummy_image()

        # 새 객체 추가
        if frame_idx % add_object_every == 0:
            new_bbox = generate_moving_bbox(
                frame_idx, tracker.next_obj_id, total_frames)
            if new_bbox:
                regions = [{'bbox': new_bbox}]
                new_obj_ids = tracker.add_new_objects(
                    frame_idx, image, regions)
                for new_obj_id in new_obj_ids:
                    object_lifetime[new_obj_id] = frame_idx

        # Dead 객체 찾기
        dead_ids = []
        for obj_id, birth_frame in list(object_lifetime.items()):
            if frame_idx - birth_frame > remove_after_frames:
                dead_ids.append(obj_id)
                object_lifetime.pop(obj_id)

        # Active 객체 리스트
        active_ids = list(object_lifetime.keys())

        # ✅ 새 파라미터 사용
        try:
            results = tracker.track(
                image,
                active_track_ids=active_ids if len(active_ids) > 0 else None,
                dead_track_ids=dead_ids if len(dead_ids) > 0 else None
            )
        except Exception as e:
            print(f"\n❌ Frame {frame_idx}에서 에러 발생: {e}")
            import traceback
            traceback.print_exc()
            break

        # 메모리 모니터링
        if frame_idx % 10 == 0:
            cpu_mem, gpu_mem = monitor.record(len(tracker.all_obj_ids))
            active_count = len(results.get('active_track_ids', []))
            removed_count = len(results.get('removed_track_ids', []))

            print(f"Frame {frame_idx:3d} | "
                  f"Total: {len(tracker.all_obj_ids):2d} | "
                  f"Active: {active_count:2d} | "
                  f"Removed: {removed_count:2d} | "
                  f"CPU: {cpu_mem:6.1f}MB | "
                  f"GPU: {gpu_mem:6.1f}MB")

    print(f"\n최종 객체 수: {len(tracker.all_obj_ids)}")
    monitor.plot("Experiment C - With Active/Dead IDs")
    return monitor


# 8. 메인 실행
if __name__ == "__main__":
    print("🔬 DAM4SAM 메모리 누수 실험 시작\n")

    try:
        # GPU 메모리 정리
        torch.cuda.empty_cache()
        time.sleep(5)

        experiment_with_active_dead_ids()

        print("\n✅ 실험 완료!")

    except Exception as e:
        print(f"\n❌ 실험 중 에러 발생: {e}")
        import traceback
        traceback.print_exc()
