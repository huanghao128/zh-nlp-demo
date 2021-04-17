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

#### keras版
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

#### tf2.0+keras版
```python
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K

class CRF(layers.Layer):
    def __init__(self, label_size):
        super(CRF, self).__init__()
        self.trans_params = tf.Variable(
            tf.random.uniform(shape=(label_size, label_size)), name="transition")
    
    @tf.function
    def call(self, inputs, labels, seq_lens):
        log_likelihood, self.trans_params = tfa.text.crf_log_likelihood(
                                                inputs, labels, seq_lens,
                                                transition_params=self.trans_params)
        loss = tf.reduce_sum(-log_likelihood)
        return loss

inputs = layers.Input(shape=(MAX_LEN,), name='input_ids', dtype='int32')
targets = layers.Input(shape=(MAX_LEN,), name='target_ids', dtype='int32')
seq_lens = layers.Input(shape=(), name='input_lens', dtype='int32')

x = layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM, mask_zero=True)(inputs)
x = layers.Bidirectional(layers.LSTM(HIDDEN_SIZE, return_sequences=True))(x)
logits = layers.Dense(CLASS_NUMS)(x)
loss = CRF(label_size=CLASS_NUMS)(logits, targets, seq_lens)

model = models.Model(inputs=[inputs, targets, seq_lens], outputs=loss)
```

### 文本情感分析
情感分析例子对应在zh_sentiment_analysis目录下，包含用tensorflow训练模型，python和Java语言分别预测的方法。

java加载模型的两种方法：
```java
// 读取tensorflow二进制的模型文件 方法1
private static Session loadTFModel(String pathname) throws IOException{
    File filename = new File(pathname);
    BufferedInputStream in = new BufferedInputStream(new FileInputStream(filename));
    ByteArrayOutputStream out = new ByteArrayOutputStream(1024);
    byte[] temp = new byte[1024];
    int size = 0;
    while((size = in.read(temp)) != -1){
        out.write(temp, 0, size);
    }
    in.close();

    byte[] graphDef = out.toByteArray();
    Graph graph = new Graph();
    graph.importGraphDef(graphDef);
    Session session = new Session(graph);
    return session;
}

// 读取tensorflow二进制的模型文件 方法2
private static Session loadTFModel2(String pathname, String tag) throws IOException{
    SavedModelBundle modelBundle = SavedModelBundle.load(pathname, tag);
    Session session = modelBundle.session();
    return session;
}
```
输入句子预测结果：
```java
// 输入模型的测试语句
int[][] sentenceBuf = getInputFromSentence(sentence, wordsMap);
Tensor inputTensor = Tensor.create(sentenceBuf);
Tensor dropProbTensor = Tensor.create(1.0f); // 预测时drop_prob = 1.0
// 输入数据，得到预测结果
Tensor result = session.runner()
        .feed("Input_Layer/input_x:0", inputTensor)
        .feed("Input_Layer/keep_prob:0", dropProbTensor)
        .fetch("Accuracy/score:0")
        .run().get(0);
```




