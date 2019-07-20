本项目是自然语言处理NLP在中文文本上的一些简单应用，如文本分类、情感分析、命名实体识别等。

### 文本分类
数据集用的是头条的标题和对应文章分类数据。数据集来自这里：

https://github.com/fate233/toutiao-text-classfication-dataset

文本分类的例子对应zh_article_classify_bilstm_attention.ipynb，这里构建的是BiLSTM+Attention的模型结构。


具体模型搭建如下：
```python
def create_classify_model(max_len, vocab_size, embedding_size, hidden_size, attention_size, class_nums):
	# 输入层
    inputs = Input(shape=(max_len,), dtype='int32')
    # Embedding层
    x = Embedding(vocab_size, embedding_size)(inputs)
    # BiLSTM层
    x = Bidirectional(LSTM(hidden_size, dropout=0.2, return_sequences=True))(x)
    # Attention层
    x = AttentionLayer(attention_size=attention_size)(x)
    # 输出层
    outputs = Dense(class_nums, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.summary() # 输出模型结构和参数数量
    return model
```

### 命名实体识别
命名实体识别的例子对应zh_ner_bilstm_crf_keras.ipynb，构建的模型是BiLSTM+CRF结构。

具体模型搭建如下：
```python
# Input输入层
inputs = Input(shape=(MAX_LEN,), dtype='int32')
# masking屏蔽层
x = Masking(mask_value=0)(inputs)
# Embedding层
x = Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True)(x)
# Bi-LSTM层
x = Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True))(x)
# Bi-LSTM展开输出
x = TimeDistributed(Dense(CLASS_NUMS))(x)
# CRF模型层
outputs = CRF(CLASS_NUMS)(x)

model = Model(inputs=inputs, outputs=outputs)
model.summary()
```

### 文本情感分析
见另一个项目：https://github.com/huanghao128/tf-sentiment-analysis
