import torch


# 모델 성능 평가 함수
def evaluate(model, test_loader, device):
    """학습 중간 모델의 정확도를 측정하는 함수입니다.

    Args:
        model : 학습에 사용되는 모델
        test_loader : 검증 단계에서 사용하는 data loader
        device : gpu, cpu 등 설정
    """
    print(f"Validation...")
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
