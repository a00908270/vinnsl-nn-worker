package at.ac.univie.a00908270.nnworker.dl4j;

import at.ac.univie.a00908270.nnworker.dataset.MnistFetcher;
import at.ac.univie.a00908270.nnworker.util.Vinnsl;
import at.ac.univie.a00908270.nnworker.vinnsl.transformation.VinnslDL4JMapper;
import at.ac.univie.a00908270.vinnsl.schema.Resultschema;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
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
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.InputStreamResource;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.StopWatch;
import org.springframework.web.client.RestTemplate;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;

public class Dl4jMnistNetworkTrainer {
	
	private static final int CLASSES_COUNT = 3;
	private static final int FEATURES_COUNT = 4;
	
/*	private static final String VINNSL_SERVICE_ENDPOINT = "http://127.0.0.1:8080/vinnsl";
	private static final String VINNSL_SERVICE_DL4J_ENDPOINT = "http://127.0.0.1:8080/dl4j";
	private static final String VINNSL_STORAGE_SERVICE_ENDPOINT = "http://127.0.0.1:8081/storage";*/
	
	private static final String VINNSL_SERVICE_ENDPOINT = "http://vinnsl-service:8080/vinnsl";
	private static final String VINNSL_SERVICE_DL4J_ENDPOINT = "http://vinnsl-service:8080/dl4j";
	private static final String VINNSL_STORAGE_SERVICE_ENDPOINT = "http://vinnsl-storage-service:8081/storage";
	
	private static final Logger log = LoggerFactory.getLogger(Dl4jMnistNetworkTrainer.class);
	
	public Dl4jMnistNetworkTrainer(Vinnsl vinnslObject) throws IOException, InterruptedException {
		
		NeuralNetConfiguration.Builder builder = VinnslDL4JMapper.INSTANCE.neuralNetConfiguration(vinnslObject);
		
		//number of rows and columns in the input pictures
		final int numRows = 28;
		final int numColumns = 28;
		int outputNum = 10; // number of output classes
		int batchSize = 128; // batch size for each epoch
		int rngSeed = 123; // random number seed for reproducibility
		int numEpochs = 15; // number of epochs to perform
		
		//download dataset if not available
		new MnistFetcher().download();
		
		//Get the DataSetIterators:
		DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
		DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);
		
		MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
				.seed(rngSeed) //include a random seed for reproducibility
				// use stochastic gradient descent as an optimization algorithm
				.updater(new Nesterovs(0.006, 0.9))
				.l2(1e-4)
				.list()
				.layer(0, new DenseLayer.Builder() //create the first, input layer with xavier initialization
						.nIn(numRows * numColumns)
						.nOut(1000)
						.activation(Activation.RELU)
						.weightInit(WeightInit.XAVIER)
						.build())
				.layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
						.nIn(1000)
						.nOut(outputNum)
						.activation(Activation.SOFTMAX)
						.weightInit(WeightInit.XAVIER)
						.build())
				.pretrain(false).backprop(true) //use backpropagation to adjust weights
				.build();
		
		
		log.info(builder.toString());
		RestTemplate restTemplate = new RestTemplate();
		restTemplate.put(String.format(VINNSL_SERVICE_DL4J_ENDPOINT + "/%s", vinnslObject.identifier), configuration.toJson());
		
		UIServer uiServer = UIServer.getInstance();
		StatsStorage statsStorage = new InMemoryStatsStorage();
		
		MultiLayerNetwork model = new MultiLayerNetwork(configuration);
		model.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(1));
		uiServer.attach(statsStorage);
		
		model.init();
		
		log.info("Train model....");
		StopWatch stopWatch = new StopWatch();
		stopWatch.start();
		
		for (int i = 0; i < numEpochs; i++) {
			model.fit(mnistTrain);
		}
		
		stopWatch.stop();
		log.info("Training took " + stopWatch.getTotalTimeSeconds());
		
		log.info("Evaluate model....");
		LinkedMultiValueMap<String, Object> map = new LinkedMultiValueMap<>();
		StringBuilder outputFileString = new StringBuilder();
		
		
		Evaluation eval = new Evaluation(outputNum); //create an evaluation object with 10 possible classes
		while (mnistTest.hasNext()) {
			DataSet next = mnistTest.next();
			INDArray output = model.output(next.getFeatures()); //get the networks prediction
			eval.eval(next.getLabels(), output); //check the prediction against the true class
			
			outputFileString.append(String.format("\nIteration #%d \n", mnistTest.cursor()));
			outputFileString.append(output.toString());
			outputFileString.append("\n");
			outputFileString.append(eval.stats());
			outputFileString.append("\n");
		}
		
		outputFileString.append("\n");
		outputFileString.append(String.format("Training took %f seconds", stopWatch.getTotalTimeSeconds()));
		outputFileString.append("\n");
		
		InputStream stream = new ByteArrayInputStream(outputFileString.toString().getBytes(StandardCharsets.UTF_8));
		
		
		map.add("file", new MultipartInputStreamFileResource(stream, "result.txt"));
		
		HttpHeaders headers = new HttpHeaders();
		headers.setContentType(MediaType.MULTIPART_FORM_DATA);
		HttpEntity<LinkedMultiValueMap<String, Object>> requestEntity = new HttpEntity<>(map, headers);
		restTemplate = new RestTemplate();
		ResponseEntity<HashMap> entity = restTemplate.postForEntity(VINNSL_STORAGE_SERVICE_ENDPOINT + "/upload", requestEntity, HashMap.class);
		log.info(entity.getBody().get("file").toString());
		
		Resultschema result = new Resultschema();
		result.setFile(entity.getBody().get("file").toString());
		
		restTemplate.put(String.format(VINNSL_SERVICE_ENDPOINT + "/%s/resultschema", vinnslObject.identifier), result);
	}
	
	class MultipartInputStreamFileResource extends InputStreamResource {
		
		private final String filename;
		
		MultipartInputStreamFileResource(InputStream inputStream, String filename) {
			super(inputStream);
			this.filename = filename;
		}
		
		@Override
		public String getFilename() {
			return this.filename;
		}
		
		@Override
		public long contentLength() throws IOException {
			return -1; // we do not want to generally read the whole stream into memory ...
		}
	}
}
