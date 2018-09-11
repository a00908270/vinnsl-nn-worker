package at.ac.univie.a00908270.nnworker.dl4j;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

public class NNDl4jFileTest {
	private static final int CLASSES_COUNT = 3;
	private static final int FEATURES_COUNT = 4;
	
	public static void main(String[] args) throws IOException, InterruptedException {
		
		
		//Define a simple MultiLayerNetwork:
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.weightInit(WeightInit.XAVIER)
				.updater(Updater.NESTEROVS)
				.learningRate(0.1)
				.list()
				.layer(0, new DenseLayer.Builder().nIn(4).nOut(3).activation(Activation.TANH).build())
				.layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX).nIn(3).nOut(3).build())
				.backprop(true).pretrain(false).build();
		
		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		
		
		//Save the model
		File locationToSave = new File("MyMultiLayerNetwork.zip");      //Where to save the network. Note: the file is in .zip format - can be opened externally
		boolean saveUpdater = true;                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
		ModelSerializer.writeModel(net, locationToSave, saveUpdater);
		
		//Load the model
		MultiLayerNetwork restored = ModelSerializer.restoreMultiLayerNetwork(locationToSave);
		
		
		System.out.println("Saved and loaded parameters are equal:      " + net.params().equals(restored.params()));
		System.out.println("Saved and loaded configurations are equal:  " + net.getLayerWiseConfigurations().equals(restored.getLayerWiseConfigurations()));
		
		
		MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new ClassPathResource("export.dl4j").getFile());
		
		
		UIServer uiServer = UIServer.getInstance();
		StatsStorage statsStorage = new InMemoryStatsStorage();
		
		DataSet allData;
		try (RecordReader recordReader = new CSVRecordReader(0, ',')) {
			recordReader.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));
			
			DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, 150, FEATURES_COUNT, CLASSES_COUNT);
			allData = iterator.next();
		}
		
		allData.shuffle(42);
		
		DataNormalization normalizer = new NormalizerStandardize();
		normalizer.fit(allData);
		normalizer.transform(allData);
		
		SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);
		DataSet trainingData = testAndTrain.getTrain();
		DataSet testData = testAndTrain.getTest();
		
/*		NeuralNetConfiguration.Builder conf = new NeuralNetConfiguration.Builder();
		conf.iterations(1000);
		conf.learningRate(0.3);
		conf.activation(Activation.SIGMOID);
		conf.layer(new DenseLayer.Builder().nIn(3).nOut(3)
				.build());*/
		
		
		/*MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
				.iterations(1000)
				.activation(Activation.TANH)
				.weightInit(WeightInit.XAVIER)
				.learningRate(0.1)
				.regularization(true).l2(0.0001)
				.list()
				.layer(0, new DenseLayer.Builder().nIn(FEATURES_COUNT).nOut(3)
						.build())
				.layer(1, new DenseLayer.Builder().nIn(3).nOut(3)
						.build())
				.layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
						.activation(Activation.SOFTMAX)
						.nIn(3).nOut(CLASSES_COUNT).build())
				.backprop(true).pretrain(false)
				.build();
		*/
		//MultiLayerNetwork model = new MultiLayerNetwork(configuration);
		model.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(1));
		uiServer.attach(statsStorage);
		
		model.init();
		
		/*File locationToSave = new File("MyMultiLayerNetwork.zip");      //Where to save the network. Note: the file is in .zip format - can be opened externally
		boolean saveUpdater = true;                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
		ModelSerializer.writeModel(model, locationToSave, saveUpdater);*/
		
		model.fit(trainingData);
		
		INDArray output = model.output(testData.getFeatureMatrix());
		System.out.println(output);
		
		
		Evaluation eval = new Evaluation(3);
		eval.eval(testData.getLabels(), output);
		System.out.println(eval.stats());
		
		
	}
}
