### 项目介绍
这是一个简单的tensorflow实现的RNN文本情感分类任务，主要是熟悉流程。

模型训练完成后Python如何调用，java如何调用。

### 模型保存
模型保存用了两种方法tf.train.write_graph和tf.saved_model.builder.SavedModelBuilder

```python
# save model1
graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["Accuracy/score"])
tf.train.write_graph(graph, ".", "./model/lstm_sentiment_model.pb", as_text=False)
# save model2
builder = tf.saved_model.builder.SavedModelBuilder('./model/lstm_sentiment_model')
builder.add_meta_graph_and_variables(sess, ["mytag"])   # java调用时需要和这里的mytag对应
builder.save()
```

### Python调用
这里用的是Python调用pb格式模型
```python
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    output_graph_def = tf.GraphDef()
    with open(model_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

    sess.run(tf.global_variables_initializer())

    input_x = sess.graph.get_tensor_by_name('Input_Layer/input_x:0')
    keep_prob = sess.graph.get_tensor_by_name('Input_Layer/keep_prob:0')
    score = sess.graph.get_tensor_by_name('Accuracy/score:0')              

    score_output = sess.run(score, feed_dict={input_x: sent2idx_new, keep_prob: 1.0})
```

### Java预测
使用java加载tensorflow模型，需要添加依赖包，项目pom.xml中添加：
```xml
<dependency>
    <groupId>org.tensorflow</groupId>
    <artifactId>tensorflow</artifactId>
    <version>1.8.0</version>
</dependency>
```
详细参考官方文档：https://www.tensorflow.org/install/lang_java?hl=zh-cn

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

