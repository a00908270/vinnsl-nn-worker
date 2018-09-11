package at.ac.univie.a00908270.nnworker.dl4j;

import at.ac.univie.a00908270.nnworker.util.Vinnsl;
import at.ac.univie.a00908270.nnworker.vinnsl.transformation.VinnslDL4JMapper;
import at.ac.univie.a00908270.vinnsl.schema.Definition;
import at.ac.univie.a00908270.vinnsl.schema.Parametervalue;
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
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.LineNumberReader;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Optional;

public class Dl4jNetworkTrainer {
	
	private static final int FEATURES_COUNT = 4;
	private static final int CLASSES_COUNT = 3;
	private static final int HIDDEN_COUNT = 3;
	private static final int LABEL_INDEX = 0;

//	private static final String VINNSL_SERVICE_ENDPOINT = "http://127.0.0.1:8080/vinnsl";
//	private static final String VINNSL_SERVICE_DL4J_ENDPOINT = "http://127.0.0.1:8080/dl4j";
//	private static final String VINNSL_STORAGE_SERVICE_ENDPOINT = "http://127.0.0.1:8081/storage";
	
	private static final String VINNSL_SERVICE_ENDPOINT = "http://vinnsl-service:8080/vinnsl";
	private static final String VINNSL_SERVICE_DL4J_ENDPOINT = "http://vinnsl-service:8080/dl4j";
	private static final String VINNSL_STORAGE_SERVICE_ENDPOINT = "http://vinnsl-storage-service:8081/storage";
	
	
	private static final Logger log = LoggerFactory.getLogger(Dl4jNetworkTrainer.class);
	
	public Dl4jNetworkTrainer(Vinnsl vinnslObject) throws IOException, InterruptedException {
		
		//initialize with parameters from vinnsl xml
		NeuralNetConfiguration.Builder builder = VinnslDL4JMapper.INSTANCE.neuralNetConfiguration(vinnslObject);
		
		int featuresCount = getFeaturesCount(vinnslObject);
		int classesCount = getClassesCount(vinnslObject);
		int hiddenCount = getHiddenCount(vinnslObject);
		
		log.info(String.format("Found features: %d, found classes: %d, found hidden classes: %d",
				featuresCount, classesCount, hiddenCount));
		
		MultiLayerConfiguration configuration = builder
				.list()
				.layer(0, new DenseLayer.Builder().nIn(featuresCount).nOut(hiddenCount)
						.build())
				.layer(1, new DenseLayer.Builder().nIn(hiddenCount).nOut(hiddenCount)
						.build())
				.layer(2, new OutputLayer.Builder()
						.activation(Activation.SOFTMAX)
						.nIn(hiddenCount).nOut(classesCount).build())
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
		
		int batchSize = getBatchSize(tmpFile);
		
		try (RecordReader recordReader = new CSVRecordReader(0, ',')) {
			recordReader.initialize(new FileSplit(tmpFile));
			
			
			DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, getLabelIndex(vinnslObject), classesCount);
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
		model.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(10));
		uiServer.attach(statsStorage);
		
		model.init();
		
		log.info("Train model....");
		StopWatch stopWatch = new StopWatch();
		stopWatch.start();
		
		model.fit(trainingData);
		
		stopWatch.stop();
		log.info("Training took " + stopWatch.getTotalTimeSeconds());
		
		INDArray output = model.output(testData.getFeatureMatrix());
		
		LinkedMultiValueMap<String, Object> map = new LinkedMultiValueMap<>();
		
		StringBuilder outputFileString = new StringBuilder();
		outputFileString.append(output.toString());
		outputFileString.append("\n");
		
		Evaluation eval = new Evaluation(classesCount);
		eval.eval(testData.getLabels(), output);
		outputFileString.append(eval.stats());
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
	
	private int getBatchSize(File tmpFile) {
		try {
			if (tmpFile.exists()) {
				
				FileReader fr = null;
				try {
					fr = new FileReader(tmpFile);
				} catch (FileNotFoundException e) {
					e.printStackTrace();
				}
				LineNumberReader lnr = new LineNumberReader(fr);
				
				int linenumber = 0;
				
				while (lnr.readLine() != null) {
					linenumber++;
				}
				
				log.info("Total number of lines : " + linenumber);
				
				lnr.close();
				
				return linenumber - 1;
				
			} else {
				log.error("File does not exists!");
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		return 0;
	}
	
	private int getFeaturesCount(Vinnsl vinnslObject) {
		
		if (vinnslObject.definition != null) {
			if (vinnslObject.definition.getStructure() != null) {
				if (vinnslObject.definition.getStructure().getInput() != null) {
					if (vinnslObject.definition.getStructure().getInput().getSize() != null) {
						return vinnslObject.definition.getStructure().getInput().getSize().intValue();
					}
				}
			}
		}
		
		return FEATURES_COUNT;
	}
	
	private int getClassesCount(Vinnsl vinnslObject) {
		
		if (vinnslObject.definition != null) {
			if (vinnslObject.definition.getStructure() != null) {
				if (vinnslObject.definition.getStructure().getOutput() != null) {
					if (vinnslObject.definition.getStructure().getOutput().getSize() != null) {
						return vinnslObject.definition.getStructure().getOutput().getSize().intValue();
					}
				}
			}
		}
		
		return CLASSES_COUNT;
	}
	
	private int getHiddenCount(Vinnsl vinnslObject) {
		if (vinnslObject.definition != null) {
			if (vinnslObject.definition.getStructure() != null) {
				if (vinnslObject.definition.getStructure().getHidden() != null) {
					Optional<Definition.Structure.Hidden> firstHidden = vinnslObject.definition.getStructure().getHidden().stream().findFirst();
					if (firstHidden.isPresent()) {
						if (firstHidden.get().getSize() != null)
							return firstHidden.get().getSize().intValue();
					}
				}
			}
		}
		
		return HIDDEN_COUNT;
	}
	
	private int getLabelIndex(Vinnsl vinnslObject) {
		Parametervalue.Valueparameter param = ((Parametervalue.Valueparameter) (vinnslObject.definition.getParameters().getValueparameterOrBoolparameterOrComboparameter().stream()
				.filter(e -> e instanceof Parametervalue.Valueparameter)
				.filter(e -> ((Parametervalue.Valueparameter) e).getName().equalsIgnoreCase("labelIndex"))
				.findFirst().orElse(null)));
		
		if (param != null && param.getValue() != null) {
			try {
				log.info("labelindex at position: %d", param.getValue().intValue());
				return param.getValue().intValue();
			} catch (NumberFormatException e) {
				log.error("labelIndex not correctly formatted");
			}
		}
		
		return LABEL_INDEX;
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
