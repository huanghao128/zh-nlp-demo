import tensorflow as tf
import numpy as np
from data_util import read_vocab
np.set_printoptions(suppress=True)

model_path = "./model/lstm_sentiment_model.pb"
vocab_path = "./data/vocab_words.txt"
special_words = ['<PAD>', '<UNK>']
max_len = 100

sentence = "手机 很 喜欢 ， 超 薄 ， 轻巧 ， 看 视频 清晰 ， 性价比 很高 ， 电池 有点 小  。"

idx2vocab, vocab2idx = read_vocab(vocab_path, special_words)
sent2idx = [vocab2idx[word] if word in vocab2idx else vocab2idx['<UNK>'] for word in sentence.split()]
sent2idx_new = np.array([[0]*(100-len(sent2idx)) + sent2idx[:100]])
print(sent2idx_new)

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
    print(score_output)