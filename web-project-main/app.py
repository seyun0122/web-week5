import os
from pathlib import Path

import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.traffic_sign_cnn import TrafficSignCNN
from traffic_sign_labels import GTSRB_LABELS, get_description

MODEL_PATH = Path("traffic_sign_cnn.pth")
device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

_model_load_error: str | None = None


def load_model():
    """Load trained weights from traffic_sign_cnn.pth. Returns None on failure."""
    global _model_load_error
    _model_load_error = None

    if not MODEL_PATH.exists():
        _model_load_error = "모델 파일을 찾을 수 없음."
        return None

    try:
        model = TrafficSignCNN(num_classes=43)
        checkpoint = torch.load(MODEL_PATH, map_location=device)

        if isinstance(checkpoint, torch.nn.Module):
            model = checkpoint
            model.to(device)
        elif isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()
        return model
    except Exception as exc:
        _model_load_error = f"모델 로딩 실패: {exc}"
        print(_model_load_error)
        return None


model = load_model()


def _preprocess(image: Image.Image) -> torch.Tensor:
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0)
    return tensor.to(device)


def predict(image):
    if image is None:
        return "이미지를 업로드해 주세요."

    if model is None:
        return (
            "## 모델 파일이 없습니다.\n\n"
            "`traffic_sign_cnn.pth` 파일이 현재 폴더에 있어야 합니다.\n\n"
            "먼저 아래 명령어로 모델을 학습하세요.\n\n"
            "```bash\n"
            "python train_gtsrb_torch.py --epochs 5\n"
            "```"
        )

    try:
        x = _preprocess(image)

        with torch.no_grad():
            outputs = model(x)
            probabilities = F.softmax(outputs, dim=1)[0]

        top_probs, top_indices = torch.topk(probabilities, k=3)

        best_idx = top_indices[0].item()
        best_label = GTSRB_LABELS[best_idx]
        best_confidence = top_probs[0].item() * 100

        lines = [
            "## 예측 결과",
            "",
            f"**{best_label}**",
            f"**설명:** {get_description(best_label)}",
            f"**신뢰도:** {best_confidence:.2f}%",
            "",
            "## Top 3 예측",
            "",
        ]

        for rank, (prob, idx) in enumerate(zip(top_probs, top_indices), start=1):
            label = GTSRB_LABELS[idx.item()]
            confidence = prob.item() * 100
            lines.append(f"{rank}. **{label}** — {confidence:.2f}%")

        lines.extend([
            "",
            "---",
            "이 결과는 학습된 이미지 분류 모델의 예측이며, 실제 도로 상황에서는 참고용으로만 사용해야 합니다.",
        ])
        return "\n".join(lines)
    except Exception as exc:
        return f"## 예측 오류\n\n처리 중 오류 발생\n\n`{exc}`"


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="교통 표지판 이미지 업로드"),
    outputs=gr.Markdown(label="분류 결과"),
    title="AI Traffic Sign Recognition",
    description=(
        "GTSRB 교통 표지판 데이터셋으로 학습한 CNN 모델을 이용해 "
        "업로드한 표지판 이미지를 43개 클래스 중 하나로 분류합니다."
    ),
    examples=None,
)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    print(f"CWD = {os.getcwd()}", flush=True)
    print(f"PORT = {port}", flush=True)
    print(f"traffic_sign_cnn.pth exists = {MODEL_PATH.exists()}", flush=True)
    print(f"MODEL LOADED = {model is not None}", flush=True)
    if _model_load_error:
        print(f"MODEL ERROR = {_model_load_error}", flush=True)
    print(f"Starting Gradio app on 0.0.0.0:{port}", flush=True)

    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
        prevent_thread_lock=False,
    )
