import torch
import numpy as np
import cv2
from PIL import Image
import psutil
import GPUtil
import time
import matplotlib.pyplot as plt
from collections import defaultdict
from tracking_wrapper_mot import DAM4SAMMOT

# 3. 메모리 모니터링 유틸리티


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

# 5. 실험 A: 객체 제거 없음 (원본 DAM4SAM)


def experiment_without_removal():
    print("=" * 60)
    print("실험 A: 객체 제거 없음 (메모리 누수 시나리오)")
    print("=" * 60)

    # DAM4SAM 초기화
    tracker = DAM4SAMMOT(model_size='tiny', checkpoint_dir='./checkpoints')
    monitor = MemoryMonitor()

    # 초기 객체
    image = generate_dummy_image()
    init_regions = [
        {'bbox': [100, 100, 100, 150]},
        {'bbox': [300, 200, 100, 150]}
    ]
    tracker.initialize(image, init_regions)

    total_frames = 500
    add_object_every = 10  # 10프레임마다 새 객체 추가

    print(f"\n총 프레임: {total_frames}")
    print(f"새 객체 추가 주기: {add_object_every} 프레임")
    print(f"예상 최종 객체 수: ~{2 + total_frames // add_object_every}개\n")

    for frame_idx in range(1, total_frames + 1):
        image = generate_dummy_image()

        # 새 객체 추가
        if frame_idx % add_object_every == 0:
            new_bbox = generate_moving_bbox(
                frame_idx, tracker.next_obj_id, total_frames)
            if new_bbox:
                region = {'bbox': new_bbox}
                tracker.add_object(image, region)

        # 추적 수행
        try:
            results = tracker.track(image)
        except Exception as e:
            print(f"\n❌ Frame {frame_idx}에서 에러 발생: {e}")
            break

        # 메모리 모니터링
        if frame_idx % 10 == 0:
            cpu_mem, gpu_mem = monitor.record(len(tracker.all_obj_ids))
            print(f"Frame {frame_idx:3d} | Objects: {len(tracker.all_obj_ids):3d} | "
                  f"CPU: {cpu_mem:6.1f}MB | GPU: {gpu_mem:6.1f}MB")

    print(f"\n최종 객체 수: {len(tracker.all_obj_ids)}")
    print(
        f"실제로 추적된 객체 수 (마지막 프레임): {len([m for m in results['masks'] if m.sum() > 0])}")

    monitor.plot("Experiment A - Without Object Removal")
    return monitor

# 6. 실험 B: 객체 명시적 제거 (수정된 버전)


def experiment_with_removal():
    print("\n" + "=" * 60)
    print("실험 B: 객체 명시적 제거 (메모리 관리 시나리오)")
    print("=" * 60)

    # DAM4SAM 초기화
    tracker = DAM4SAMMOT(model_size='tiny', checkpoint_dir='./checkpoints')
    monitor = MemoryMonitor()

    # 초기 객체
    image = generate_dummy_image()
    init_regions = [
        {'bbox': [100, 100, 100, 150]},
        {'bbox': [300, 200, 100, 150]}
    ]
    tracker.initialize(image, init_regions)

    total_frames = 500
    add_object_every = 10
    remove_after_frames = 50  # 객체를 50프레임 후 제거

    object_lifetime = {}  # {obj_id: first_frame}

    print(f"\n총 프레임: {total_frames}")
    print(f"새 객체 추가 주기: {add_object_every} 프레임")
    print(f"객체 수명: {remove_after_frames} 프레임\n")

    for frame_idx in range(1, total_frames + 1):
        image = generate_dummy_image()

        # 새 객체 추가
        if frame_idx % add_object_every == 0:
            new_bbox = generate_moving_bbox(
                frame_idx, tracker.next_obj_id, total_frames)
            if new_bbox:
                region = {'bbox': new_bbox}
                new_obj_id, _ = tracker.add_object(image, region)
                object_lifetime[new_obj_id] = frame_idx

        # 오래된 객체 제거
        to_remove = []
        for obj_id, birth_frame in object_lifetime.items():
            if frame_idx - birth_frame > remove_after_frames:
                to_remove.append(obj_id)

        for obj_id in to_remove:
            if obj_id in tracker.all_obj_ids:
                # 명시적 제거
                tracker.all_obj_ids.remove(obj_id)
                tracker.per_object_outputs_all.pop(obj_id, None)
                tracker.per_object_obj_ptr.pop(obj_id, None)
                tracker.add_to_drm_next.pop(obj_id, None)

                # 인덱스 기반 리스트도 정리 (obj_id가 인덱스라 가정)
                # 실제로는 obj_id → index 매핑 필요
                object_lifetime.pop(obj_id)

        # 추적 수행
        try:
            results = tracker.track(image)
        except Exception as e:
            print(f"\n❌ Frame {frame_idx}에서 에러 발생: {e}")
            break

        # 메모리 모니터링
        if frame_idx % 10 == 0:
            cpu_mem, gpu_mem = monitor.record(len(tracker.all_obj_ids))
            print(f"Frame {frame_idx:3d} | Objects: {len(tracker.all_obj_ids):3d} | "
                  f"CPU: {cpu_mem:6.1f}MB | GPU: {gpu_mem:6.1f}MB")

    print(f"\n최종 객체 수: {len(tracker.all_obj_ids)}")

    monitor.plot("Experiment B - With Object Removal")
    return monitor

# 7. 비교 플롯


def compare_experiments(monitor_a, monitor_b):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # CPU 메모리 비교
    axes[0, 0].plot(monitor_a.timestamps, monitor_a.cpu_memory,
                    label='Without Removal', marker='o', linewidth=2)
    axes[0, 0].plot(monitor_b.timestamps, monitor_b.cpu_memory,
                    label='With Removal', marker='s', linewidth=2)
    axes[0, 0].set_xlabel('Time (seconds)')
    axes[0, 0].set_ylabel('CPU Memory (MB)')
    axes[0, 0].set_title('CPU Memory Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # GPU 메모리 비교
    axes[0, 1].plot(monitor_a.timestamps, monitor_a.gpu_memory,
                    label='Without Removal', marker='o', linewidth=2)
    axes[0, 1].plot(monitor_b.timestamps, monitor_b.gpu_memory,
                    label='With Removal', marker='s', linewidth=2)
    axes[0, 1].set_xlabel('Time (seconds)')
    axes[0, 1].set_ylabel('GPU Memory (MB)')
    axes[0, 1].set_title('GPU Memory Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 객체 수 비교
    axes[1, 0].plot(monitor_a.timestamps, monitor_a.num_objects,
                    label='Without Removal', marker='o', linewidth=2, color='red')
    axes[1, 0].plot(monitor_b.timestamps, monitor_b.num_objects,
                    label='With Removal', marker='s', linewidth=2, color='green')
    axes[1, 0].set_xlabel('Time (seconds)')
    axes[1, 0].set_ylabel('Number of Objects')
    axes[1, 0].set_title('Object Count Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 통계 요약
    stats_text = f"""
    Experiment A (Without Removal):
    - Peak CPU: {max(monitor_a.cpu_memory):.1f} MB
    - Peak GPU: {max(monitor_a.gpu_memory):.1f} MB
    - Max Objects: {max(monitor_a.num_objects)}
    
    Experiment B (With Removal):
    - Peak CPU: {max(monitor_b.cpu_memory):.1f} MB
    - Peak GPU: {max(monitor_b.gpu_memory):.1f} MB
    - Max Objects: {max(monitor_b.num_objects)}
    
    Memory Saved:
    - CPU: {max(monitor_a.cpu_memory) - max(monitor_b.cpu_memory):.1f} MB
    - GPU: {max(monitor_a.gpu_memory) - max(monitor_b.gpu_memory):.1f} MB
    """

    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                    verticalalignment='center', transform=axes[1, 1].transAxes)
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('Memory_Comparison.png', dpi=150)
    plt.show()


# 8. 메인 실행
if __name__ == "__main__":
    print("🔬 DAM4SAM 메모리 누수 실험 시작\n")

    # 실험 A 실행
    monitor_a = experiment_without_removal()

    # GPU 메모리 정리
    torch.cuda.empty_cache()
    time.sleep(5)

    # 실험 B 실행
    monitor_b = experiment_with_removal()

    # 비교 플롯
    compare_experiments(monitor_a, monitor_b)

    print("\n✅ 실험 완료!")
