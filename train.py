# 필요한 라이브러리 import
import torch
import torch.nn as nn
import numpy as np
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model.vgg import VGG11


# 하이퍼 파라미터 정의 (epoch, lr, etc...)
batch_size = 100
num_classes = 10
num_epochs = 10
lr = 0.0001

device = "cuda" if torch.cuda.is_available() else "cpu"

# 데이터 load (Dataset)
## 데이터 전처리
## cifar10 이미지의 평균과 표준편차
CIFAR_MEAN = [0.491, 0.482, 0.447]
CIFAR_STD = [0.247, 0.244, 0.262]

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)]
)

train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
test_datasets = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

# 데이터 로더 정의 (DataLoader)
train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
)
test_loader = DataLoader(
    dataset=test_datasets, batch_size=batch_size, shuffle=False, num_workers=0
)

# 모델 선언
model = VGG11(num_classes=num_classes).to(device)

# loss, optimizer 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


def evaluate(model, test_loader):
    print(f"Test data를 사용해 모델의 성능을 측정합니다.")
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return correct / total


best_acc = 0
for epoch in range(num_epochs):
    model.train()

    # (for) 데이터 불러오기
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 불러온 데이터 모델에 넣기
        outputs = model(images)  # [batch, 10]
        loss = criterion(outputs, labels)

        # 출력을 바탕으로 loss 계산
        # loss로 back propagation
        # optimizer로 최적화 진행
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 학습 중간 결과로 성능을 평가
        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch + 1} / {num_epochs}], Step [{i + 1} / {len(train_loader)}], Loss : {loss.item()}"
            )

            acc = evaluate(model, test_loader=test_loader)
            # 성능이 만족스럽다면 weight 저장
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), f"./best_model.ckpt")
