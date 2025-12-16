1. 프로젝트 개요

본 프로젝트는 가림(Occlusion) 및 교차(Crossing) 상황에서 빈번하게 발생하는 ID Switching 문제를 해결 하기 위한 깊이 인식 기반 다중 객체 추적(Multi-Object Tracking, MOT) 파이프라인을 제안한다.

DAST(Depth-Aware Segmentation Tracker)는  
기존 2D 기반 Tracking-by-Detection 방식의 한계를 극복하기 위해  
단안 RGB 영상으로부터 깊이 정보를 추정하여 가상 3D(Pseudo-3D) 공간을 구성하고,  
이를 기반으로 3D 추적 + 세그멘테이션 기반 이중 검증 구조를 결합한 End-to-End 자동화 시스템이다.

2. 핵심 특징

- 2D RGB 영상만 사용 (LiDAR / RGB-D 센서 불필요)
- ID Switching 대폭 감소 (MOT15 기준 약 72%)
- 완전 자동화 파이프라인 (수동 프롬프트 불필요)
- 3D 추적 + 세그멘테이션 기반 이중 검증
- 군중 환경(Occlusion, Crossing)에 강건

3. 전체 파이프라인 구조
![아키텍처1](https://github.com/user-attachments/assets/cb3967dc-4ed9-4bfc-95c8-e399a4179655)

입력 영상 (RGB)
↓
YOLO11n / YOLO11n-seg (객체 탐지 및 초기화)
↓
ZoeDepth (단안 깊이 추정)
↓
3D 정규화 (Pseudo-3D 생성)
↓
HybridTrack (3D 다중 객체 추적 및 ID 관리)
↔
DAM4SAM (세그멘테이션 기반 추적 및 이중 검증)

4. 디렉토리 구조

<img width="607" height="978" alt="image" src="https://github.com/user-attachments/assets/a212f96e-4abe-423f-b39c-09af7272b9a0" />


5. 실행 환경 설정

5.1 Python 환경

- Python 3.9 ~ 3.10 권장
- CUDA 지원 GPU 사용 권장

conda create -n dast python=3.10
conda activate dast

5.2 라이브러리 설치

pip install -r requirements.txt

6. 실행 방법

6.1 Step-by-Step 실행 (권장)

전체 실행 흐름은 아래 노트북을 기준으로 구성되어 있다.

jupyter notebook src/try_whole_model.ipynb


노트북 실행 단계:

1. YOLO11 기반 객체 탐지
2. ZoeDepth 기반 Depth Map 생성
3. 3D 좌표 정규화 (Pseudo-3D)
4. HybridTrack 기반 3D 다중 객체 추적
5. DAM4SAM 기반 세그멘테이션 추적
6. 이중 검증을 통한 ID Switching 억제
7. 결과 시각화 및 저장

6.2 전체 파이프라인 실행 (스크립트)

python src/run_tracking.py

7. 결과 출력

실행 결과는 `results/` 디렉토리에 저장된다.

* 추적 결과 영상
* 프레임별 추적 로그
* ID Switching 통계
* MOT 형식 결과 파일

8. 성능 요약

* MOT15 기준 ID Switching 약 72% 감소
* Occlusion 및 Crossing 상황에서 ID 일관성 유지
* 기존 2D 추적기 대비 안정적인 장기 추적 가능

9. 활용 분야

* 지능형 CCTV 및 보안 관제
* 스마트 시티 교통 분석
* 로봇 비전 기반 타겟 추적
* 스포츠 영상 분석
* 리테일 고객 동선 분석

10. 참고 사항

* 본 프로젝트는 연구 및 교육 목적의 오픈소스 프로젝트이다.
* YOLO, ZoeDepth, HybridTrack, DAM4SAM의 라이선스를 각각 준수해야 한다.

11. 팀 구성

공개 SW 프로젝트 4조

* 이준호: DAST 아키텍처 설계, 3D 추적
* 이재필: 깊이 추정 및 3D 정규화
* 이수: YOLO 기반 탐지 자동화
* 김동명: DAM4SAM 메모리 최적화
* 김재홍: 파이프라인 통합 및 성능 평가


