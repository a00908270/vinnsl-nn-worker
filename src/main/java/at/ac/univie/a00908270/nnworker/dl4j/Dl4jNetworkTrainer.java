package at.ac.univie.a00908270.nnworker.dl4j;

import at.ac.univie.a00908270.nnworker.util.Vinnsl;
import at.ac.univie.a00908270.nnworker.vinnsl.transformation.VinnslDL4JMapper;
import at.ac.univie.a00908270.vinnsl.schema.Resultschema;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class Dl4jNetworkTrainer {
	
	private static final int CLASSES_COUNT = 3;
	private static final int FEATURES_COUNT = 4;
	
	private static final Logger log = LoggerFactory.getLogger(Dl4jNetworkTrainer.class);
	
	public Dl4jNetworkTrainer(Vinnsl vinnslObject) throws IOException, InterruptedException {
		
		NeuralNetConfiguration.Builder builder = VinnslDL4JMapper.INSTANCE.neuralNetConfiguration(vinnslObject);
		log.info(builder.toString());
		
		MultiLayerConfiguration configuration = builder
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
		
		MultiLayerNetwork model = new MultiLayerNetwork(configuration);
		model.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(1));
		uiServer.attach(statsStorage);
		
		model.init();
		
		model.fit(trainingData);
		
		INDArray output = model.output(testData.getFeatureMatrix());
		
		vinnslObject.result.setTable(new Resultschema.Table().getInputAndOutput().add(output.toString()));
		
		log.info(output.toString());
		
	}
}
