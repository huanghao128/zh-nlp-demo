
import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import org.tensorflow.Graph;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

public class predictModel {
    private static String TensorFlow_MODEL_PATH = "../data/lstm_sentiment_model.pb";
    private static String TensorFlow_MODEL_PATH2 = "../data/lstm_sentiment_model/";
    private static String WORD_INDEX_PATH = "../data/vocab_words.txt";
    private static int MAX_SEQUENCE_LENGTH = 100;
    private static int CLASS_NUM = 2;

    public static void main(String[] args) throws IOException{
        // 构建词典Map
        String[] special_words = {"<PAD>", "<UNK>"};
        Map<String, Integer> wordsMap = readVocabFile(WORD_INDEX_PATH, special_words);
        System.out.println("vocabulary size:"+wordsMap.size());

//        // 加载TF训练好的模型 方法1
//        Session session = loadTFModel(TensorFlow_MODEL_PATH);

         // 加载TF训练好的模型 方法2
         Session session = loadTFModel2(TensorFlow_MODEL_PATH2, "mytag");

        // 输入预测的标题
        String test_sentence = "手机 很 喜欢 ， 超 薄 ， 轻巧 ， 看 视频 清晰 ， 性价比 很高 ，电池 有点 小  。";
        System.out.println("sentence: "+test_sentence);

        // 预测结果
        float[][] resultArray = predict_sentence(session, wordsMap, test_sentence);
        System.out.println("Predict: " + resultArray[0][0]+" "+resultArray[0][1]);
    }

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

    // 读取词典文件
    private static Map<String, Integer> readVocabFile(String pathname, String[] special_words) throws IOException{
        Map<String, Integer> wordMap = new HashMap<>();
        wordMap.put(special_words[0], 0);
        wordMap.put(special_words[1], 1);

        File filename = new File(pathname);
        InputStreamReader reader = new InputStreamReader(new FileInputStream(filename));
        BufferedReader br = new BufferedReader(reader);
        String line = br.readLine();

        int count = 2;
        while(line != null){
            wordMap.put(line, count++);
            line = br.readLine();
        }
        return wordMap;
    }


    // 读取分词后的一个样本，并建立索引
    public static int[][] getInputFromSentence(String sentence, Map<String, Integer> wordIndexMap) {
        int[][] indexArray = new int[1][MAX_SEQUENCE_LENGTH];
        String[] words = sentence.split(" ");

        if(words.length >= 100){
            for(int i = 0; i < MAX_SEQUENCE_LENGTH; i++){
                if(wordIndexMap.containsKey(words[i])){
                    indexArray[0][i] = wordIndexMap.get(words[i]);
                }else{
                    indexArray[0][i] = wordIndexMap.get("<UNK>");
                }
            }
        }else{
            for(int i = 0; i < words.length; i++){
                if(wordIndexMap.containsKey(words[i])){
                    indexArray[0][MAX_SEQUENCE_LENGTH-words.length+i] = wordIndexMap.get(words[i]);
                }else{
                    indexArray[0][MAX_SEQUENCE_LENGTH-words.length+i] = wordIndexMap.get("<UNK>");
                }
            }
        }
        return indexArray;
    }

    /**
     *
     * @param session Tensorflow的session会话
     * @param wordsMap 词典对应关系
     * @param sentence 输入要预测的句子
     * @return
     */
    private static float[][] predict_sentence(Session session, Map<String, Integer> wordsMap, String sentence){
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

        long[] rshape = result.shape();
        int batchSize = (int) rshape[0];
        float[][] resultArray = new float[batchSize][CLASS_NUM];
        result.copyTo(resultArray); // 输出结果Tensor复制到二维数组中

        return resultArray;
    }
}
