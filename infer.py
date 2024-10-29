# 패키지를 임포트
import torch
from PIL import Image
from torchvision import transforms
from torch.nn.functional import softmax

from model.vgg import VGG11


device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = 10

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
model = VGG11(num_classes=num_classes)

# 속이 빈 모델에 학습된 모델의 wieght를 덮어 씌우기
weight = torch.load("best_model.ckpt", map_location=device)
model.load_state_dict(weight)
model = model.to(device)

# 준비된 데이터를 모델에 넣기
output = model(image)

## 결과를 분석
# 결과를 사람이 이해할 수 있는 형태로 변환
# 모델의 추론 결과를 보고 객관적인 평가 내려보기
probability = softmax(output, dim=1)
values, indices = torch.max(probability, dim=1)
prob = values.item() * 100
predict = indices.item()

print(f"해당 모델은 이미지를 보고 {prob:.2f}% 의 확률로 {predict}이라고 대답했습니다.")
