# 필요한 라이브러리 import
import torch


# 하이퍼 파라미터 정의 (epoch, lr, etc...)
batch_size = 100
num_classes = 10
epoch = 10
lr = 0.0001

device = "cuda" if torch.cuda.is_available() else "cpu"

# 데이터 load (Dataset)
## 데이터 전처리

# 데이터 로더 정의 (DataLoader)

# 모델 선언

# loss, optimizer 정의

# (for) 데이터 불러오기
# 불러온 데이터 모델에 넣기

# 출력을 바탕으로 loss 계산
# loss로 back propagation
# optimizer로 최적화 진행


# 학습 중간 결과로 성능을 평가
# 성능이 만족스럽다면 weight 저장
