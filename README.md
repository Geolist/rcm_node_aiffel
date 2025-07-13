<<<<<<< HEAD
## 가상환경 설정

* 새로운 가상환경을 생성한 뒤, 아래 버전으로 진행해 주세요.
  (권장: `Python 3.11`, `TensorFlow 2.15.0`)

```bash
pip install tensorflow
pip install streamlit numpy pandas joblib scikit-learn tqdm
```

## 실행 방법

* 프로젝트 폴더 안에서 다음 명령어로 실행해 주세요.

  * `show_st.py`: 노드 테스트 코드
  * `show_st2.py`: 프로젝트 전체 실행 코드

```bash
streamlit run show_st.py
```

## 노트북 코드 설명

* 하나는 **모델이 정상적으로 로드되는지 확인**하는 코드입니다.

  > 바로 Streamlit으로 실행하면 에러가 생겼을 때 디버깅이 번거로울 수 있으므로, 먼저 이 코드로 확인하세요.

* 다른 하나는 **새로운 모델을 학습시키고 가중치를 저장하는 코드**입니다.

  > 이미 학습된 가중치 파일도 함께 전달드렸지만, 새 모델을 만들 때는 해당 코드를 참고하시기 바랍니다.

## 추가로 진행해야 할 작업

* 모델의 **성능 향상을 위한 다양한 시도**를 자유롭게 해보세요!

## 주의사항 및 참고

* **TensorFlow 버전에 따라** 가중치 파일의 확장자가 다를 수 있습니다. (제공되는 코드는 .weights 가 하나씩 더 붙어있습니다)참고해주세요
* 코드 내 `dtype.longlong` 부분은 `int64` 또는 `int32`로 변경되어야 할 수 있습니다. 참고해주세요
* 다양한 오류가 발생하지만 참고 해주세요. 😄
=======
# rcm_node_aiffel

노드상의 조건으로 mymodel ndcg : 0.66161mymodel hitrate : 0.63017

파라미터(조건) 변경 :
epochs=10
learning_rate= 0.0005
dropout= 0.3
batch_size = 1024
embed_dim= 32

# 모델 정의
autoIntMLP_model = AutoIntMLPModel(
    field_dims=field_dims,
    embedding_size=embed_dim,
    att_layer_num=3,
    att_head_num=2,
    att_res=True,
    dnn_hidden_units=(64, 32),               # 추가: DNN 은닉층 구조
    dnn_activation='swish',                  # 추가: 활성화 함수
    l2_reg_dnn=1e-4,
    l2_reg_embedding=1e-5,
    dnn_use_bn=True,
    dnn_dropout=dropout,
    init_std=0.001
)
>>>>>>> 8569a50400bc5aeca73e7f9d02bd730cf6a6dd9f
