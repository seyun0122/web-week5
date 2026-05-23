# AI Traffic Sign Recognition

GTSRB 교통 표지판 데이터셋을 활용하여 교통 표지판 이미지를 분류하는 인공지능 웹앱이다.  
PyTorch로 CNN 모델을 학습하고, Gradio를 이용해 사용자가 이미지를 업로드하면 표지판 종류와 신뢰도를 확인할 수 있도록 만들었다.

## 프로젝트 구조

```text
app.py                      
train_gtsrb_torch.py        
traffic_sign_labels.py      
traffic_sign_cnn.pth        
models/
  traffic_sign_cnn.py
requirements.txt
.python-version
render.yaml
```

## 로컬 실행

```bash
pip install -r requirements.txt
python app.py
```

브라우저에서 Gradio 주소(기본 `http://127.0.0.1:7860`)를 열고 교통 표지판 이미지를 업로드하면 된다.

```bash
python train_gtsrb_torch.py --epochs 5
```

학습이 끝나면 `traffic_sign_cnn.pth` 파일이 생성된다.

3. **Start Command:**

   ```bash
   python app.py
   ```

4. Python 버전은 `.python-version`(3.11.9) 또는 `render.yaml`의 `PYTHON_VERSION`을 사용한다.

`render.yaml` Blueprint를 사용하는 경우, 저장소의 Blueprint 설정으로 동일하게 배포할 수 있다.

Render의 Start Command는 아래처럼 설정하면 된다.

```bash
python app.py
```
주의할 점은 `traffic_sign_cnn.pth` 파일이 GitHub 저장소에 같이 올라가 있어야 Render에서도 바로 예측할 수 있다는 점이다.  
모델 파일이 없다면 Render에서는 학습 코드가 아니라 웹앱만 실행되므로 예측이 되지 않는다.

