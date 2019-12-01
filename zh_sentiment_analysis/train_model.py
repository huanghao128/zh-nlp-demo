import tensorflow as tf
import keras
from keras.preprocessing import sequence
from data_util import read_vocab, read_dataset, batch_generator
from module import RNNSentimentModel
tf.reset_default_graph()

data_path = "./data/zh_sentiment_dataset_seg.txt"
vocab_path = "./data/vocab_words.txt"
special_words = ['<PAD>', '<UNK>']

# 加载词典和数据集
idx2vocab, vocab2idx = read_vocab(vocab_path, special_words)
all_datas, all_labels = read_dataset(data_path, vocab2idx)

vocab_size = len(vocab2idx)
max_seq_length = 100
embedding_size = 100
hidden_size = 128
layer_size = 2
n_class = 2
drop_keep_prob = 0.5
learning_rate = 0.001
batch_size = 64
epochs = 10

# --------------------padding and split data ------------------

count = len(all_labels)
# 数据集划分比例
rate1, rate2 = 0.8, 0.9  # train-0.8, test-0.1, dev-0.1
# 数据的填充，不够长度左侧padding，大于长度右侧截断
new_datas = sequence.pad_sequences(all_datas, maxlen=max_seq_length, padding='pre', truncating='post')
# 类别one-hot化
new_labels = keras.utils.to_categorical(all_labels, n_class)
# 根据比例划分训练集、测试集、验证集
x_train, y_train = new_datas[:int(count*rate1)], new_labels[:int(count*rate1)]
x_test, y_test = new_datas[int(count*rate1):int(count*rate2)], new_labels[int(count*rate1):int(count*rate2)]
x_val, y_val = new_datas[int(count*rate2):], new_labels[int(count*rate2):]


# -----------------train and test model------------------

model = RNNSentimentModel(max_seq_length=max_seq_length,
                          vocab_size=vocab_size,
                          embedding_size=embedding_size,
                          hidden_size=hidden_size,
                          layer_size=layer_size,
                          n_class=n_class,
                          learning_rate=learning_rate)

saver = tf.train.Saver()
# training model
with tf.Session() as sess:
    # init variables
    sess.run(tf.global_variables_initializer())
    # Batch generators
    train_batch_generator = batch_generator([x_train, y_train], batch_size)
    test_batch_generator = batch_generator([x_test, y_test], batch_size)
    dev_batch_generator = batch_generator([x_val, y_val], batch_size)

    print("Start training...")
    for epoch in range(epochs):
        loss_train, loss_test, accuracy_train, accuracy_test = 0, 0, 0, 0
        num_batches = x_train.shape[0] // batch_size
        for i in range(0, num_batches):
            x_batch, y_batch = next(train_batch_generator)
            loss_batch, train_acc, _ = sess.run([model.loss, model.accuracy, model.optimizer], feed_dict={
                model.input_x: x_batch, model.label_y: y_batch, model.keep_prob: drop_keep_prob})
            print("\rEpoch: {:d} batch: {:d}/{:d} loss: {:.4f} acc: {:.4f} | {:.2%}".format(
                epoch + 1, i + 1, num_batches, loss_batch, train_acc, (i + 1) * 1.0 / num_batches), end='')
            accuracy_train += train_acc
            loss_train += loss_batch
        loss_train /= num_batches
        accuracy_train /= num_batches

        # test model
        num_batches = x_test.shape[0] // batch_size
        for i in range(0, num_batches):
            x_batch, y_batch = next(test_batch_generator)
            loss_batch, test_acc = sess.run([model.loss, model.accuracy], feed_dict={
                model.input_x: x_batch, model.label_y: y_batch, model.keep_prob: 1.0})
            accuracy_test += test_acc
            loss_test += loss_batch
        accuracy_test /= num_batches
        loss_test /= num_batches
        print("\rEpoch: {:d}/{:d} train_loss: {:.4f} test_loss: {:.4f} train_acc: {:.4f} test_acc: {:.4f}".format(
            epoch + 1, epochs, loss_train, loss_test, accuracy_train, accuracy_test), end='\n')

    # dev model
    loss_dev, accuracy_dev = 0, 0
    num_batches = x_val.shape[0] // batch_size
    for i in range(0, num_batches):
        x_batch, y_batch = next(dev_batch_generator)
        loss_dev_batch, dev_acc = sess.run([model.loss, model.accuracy], feed_dict={
            model.input_x: x_batch, model.label_y: y_batch, model.keep_prob: 1.0})
        accuracy_dev += dev_acc
        loss_dev += loss_dev_batch
    accuracy_dev /= num_batches
    loss_dev /= num_batches
    print("dev_loss: {:.4f}, dev_acc: {:.4f}".format(loss_dev, accuracy_dev))

    # save model1
    graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["Accuracy/score"])
    tf.train.write_graph(graph, ".", "./model/lstm_sentiment_model.pb", as_text=False)
    # save model2
    builder = tf.saved_model.builder.SavedModelBuilder('./model/lstm_sentiment_model')
    builder.add_meta_graph_and_variables(sess, ["mytag"])   # java调用时需要和这里的mytag对应
    builder.save()
