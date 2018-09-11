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
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.io.File;

//This version classifies entries into 2 classes
// 0 if score <=90
// 1 if score >90
public class Wine_2Classes {
	
	//Main function. This is what should be run
	public static void main(String[] args) throws Exception {
		//seed for RNG. used for reproducibility
		int seed = 123;
		double learningRate = 0.005;
		int batchSize = 1000;
		int nEpochs = 50;
		
		//number of non-label columns in data set
		int numInputs = 2;
		//number of classes that can be output
		int numOutputs = 2;
		int numHiddenNodes = 30;
		
		//Data sets
		final String filenameTrain = new ClassPathResource("winemag-data-SET_2-TRAIN.csv").getFile().getPath();
		final String filenameTest = new ClassPathResource("winemag-data-SET_2-TEST.csv").getFile().getPath();
		
		//Load the training data:
		RecordReader rr = new CSVRecordReader();
		rr.initialize(new FileSplit(new File(filenameTrain)));
		DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, 0, 3);
		
		//Load the test/evaluation data:
		RecordReader rrTest = new CSVRecordReader();
		rrTest.initialize(new FileSplit(new File(filenameTest)));
		DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 0, 3);
		
		//Configure the NN
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(seed)
				.updater(new Nesterovs(learningRate, 0.9))
				.list()
				.layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
						.weightInit(WeightInit.XAVIER)
						.activation(Activation.RELU)
						.build())
				.layer(1, new DenseLayer.Builder()
						.weightInit(WeightInit.XAVIER)
						.activation(Activation.RELU)
						.nIn(numHiddenNodes).nOut(numHiddenNodes).build())
				.layer(2, new DenseLayer.Builder()
						.weightInit(WeightInit.XAVIER)
						.activation(Activation.RELU)
						.nIn(numHiddenNodes).nOut(numHiddenNodes).build())
				.layer(3, new DenseLayer.Builder()
						.weightInit(WeightInit.XAVIER)
						.activation(Activation.RELU)
						.nIn(numHiddenNodes).nOut(numHiddenNodes).build())
				.layer(4, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
						.weightInit(WeightInit.XAVIER)
						.activation(Activation.SOFTMAX).weightInit(WeightInit.XAVIER)
						.nIn(numHiddenNodes).nOut(numOutputs).build())
				.pretrain(false).backprop(true).build();
		
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		
		//Initialize the user interface backend
		UIServer uiServer = UIServer.getInstance();
		
		//Configure where the network information (gradients, activations, score vs. time etc) is to be stored
		StatsStorage statsStorage = new InMemoryStatsStorage();
		
		//Print score every 50 parameter updates
		//Add the StatsListener to collect this information from the network, as it trains
		model.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(50));
		
		
		for (int n = 0; n < nEpochs; n++) {
			model.fit(trainIter);
		}
		
		uiServer.attach(statsStorage);
		
		System.out.println("Evaluate model....");
		Evaluation eval = new Evaluation(numOutputs);
		while (testIter.hasNext()) {
			DataSet t = testIter.next();
			INDArray features = t.getFeatureMatrix();
			INDArray lables = t.getLabels();
			INDArray predicted = model.output(features, false);
			
			eval.eval(lables, predicted);
			
		}
		
		//Print the evaluation statistics
		System.out.println(eval.stats());
		
		
		System.out.println("Complete");
	}
}
