package at.ac.univie.a00908270.nnworker.dl4j;

import at.ac.univie.a00908270.nnworker.util.Vinnsl;
import at.ac.univie.a00908270.nnworker.vinnsl.transformation.VinnslDL4JMapper;
import at.ac.univie.a00908270.vinnsl.schema.Resultschema;
import org.apache.commons.io.FileUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.InputStreamResource;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.StopWatch;
import org.springframework.web.client.RestTemplate;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;

public class Dl4jNetworkTrainer {
	
	private static final int CLASSES_COUNT = 3;
	private static final int FEATURES_COUNT = 4;

//	private static final String VINNSL_SERVICE_ENDPOINT = "http://127.0.0.1:8080/vinnsl";
//	private static final String VINNSL_SERVICE_DL4J_ENDPOINT = "http://127.0.0.1:8080/dl4j";
//	private static final String VINNSL_STORAGE_SERVICE_ENDPOINT = "http://127.0.0.1:8081/storage";
	
	private static final String VINNSL_SERVICE_ENDPOINT = "http://vinnsl-service:8080/vinnsl";
	private static final String VINNSL_SERVICE_DL4J_ENDPOINT = "http://vinnsl-service:8080/dl4j";
	private static final String VINNSL_STORAGE_SERVICE_ENDPOINT = "http://vinnsl-storage-service:8081/storage";
	
	
	private static final Logger log = LoggerFactory.getLogger(Dl4jNetworkTrainer.class);
	
	public Dl4jNetworkTrainer(Vinnsl vinnslObject) throws IOException, InterruptedException {
		
		NeuralNetConfiguration.Builder builder = VinnslDL4JMapper.INSTANCE.neuralNetConfiguration(vinnslObject);
		
		MultiLayerConfiguration configuration = builder
				.list()
				.layer(0, new DenseLayer.Builder().nIn(FEATURES_COUNT).nOut(3)
						.build())
				.layer(1, new DenseLayer.Builder().nIn(3).nOut(3)
						.build())
				.layer(2, new OutputLayer.Builder()
						.activation(Activation.SOFTMAX)
						.nIn(3).nOut(CLASSES_COUNT).build())
				.backprop(true).pretrain(false)
				.build();
		
		log.info(builder.toString());
		RestTemplate restTemplate = new RestTemplate();
		restTemplate.put(String.format(VINNSL_SERVICE_DL4J_ENDPOINT + "/%s", vinnslObject.identifier), configuration.toJson());
		
		UIServer uiServer = UIServer.getInstance();
		StatsStorage statsStorage = new InMemoryStatsStorage();
		
		DataSet allData;
		
		
		ResponseEntity<byte[]> response = restTemplate.getForEntity(
				String.format(VINNSL_STORAGE_SERVICE_ENDPOINT + "/files/%s", vinnslObject.definition.getData().getDataSchemaID()),
				byte[].class);
		
		File tmpFile = File.createTempFile("iris", "txt");
		if (response.getStatusCode() == HttpStatus.OK) {
			
			FileUtils.writeByteArrayToFile(tmpFile, response.getBody());
		}
		
		try (RecordReader recordReader = new CSVRecordReader(0, ',')) {
			recordReader.initialize(new FileSplit(tmpFile));
			//recordReader.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));
			
			DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, 150, FEATURES_COUNT, CLASSES_COUNT);
			allData = iterator.next();
		} finally {
			tmpFile.delete();
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
		
		log.info("Train model....");
		StopWatch stopWatch = new StopWatch();
		stopWatch.start();
		
		model.fit(trainingData);
		
		stopWatch.stop();
		log.info("Taining took " + stopWatch.getTotalTimeSeconds());
		
		INDArray output = model.output(testData.getFeatureMatrix());
		
		LinkedMultiValueMap<String, Object> map = new LinkedMultiValueMap<>();
		
		StringBuilder outputFileString = new StringBuilder();
		outputFileString.append(output.toString());
		outputFileString.append("\n");
		
		Evaluation eval = new Evaluation(3);
		eval.eval(testData.getLabels(), output);
		outputFileString.append(eval.stats());
		outputFileString.append("\n");
		outputFileString.append("Training took " + stopWatch.getTotalTimeSeconds());
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
