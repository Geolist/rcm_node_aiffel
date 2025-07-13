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
