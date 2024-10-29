# 패키지를 임포트
import torch
from PIL import Image
from torchvision import transforms


device = "cuda" if torch.cuda.is_available() else "cpu"

# 추론에 사용할 데이터를 준비 (전처리)
image_path = "test_image.jpeg"
image = Image.open(image_path)
image = image.resize((32, 32))

CIFAR_MEAN = [0.491, 0.482, 0.447]
CIFAR_STD = [0.247, 0.244, 0.262]

transform = transforms.Compose[
    transforms.ToTensor(), transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)
]

image = transform(image)
image = image.unsqueeze(0).to(device)

## 학습이 완료된 최고의 모델을 준비하기
# 설계도 + Hparam 모델 껍대기 만들기
# 속이 빈 모델에 학습된 모델의 wieght를 덮어 씌우기

# 준비된 데이터를 모델에 넣기

## 결과를 분석
# 결과를 사람이 이해할 수 있는 형태로 변환
# 모델의 추론 결과를 보고 객관적인 평가 내려보기
