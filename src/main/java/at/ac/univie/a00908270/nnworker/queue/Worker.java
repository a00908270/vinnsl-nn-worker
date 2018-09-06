package at.ac.univie.a00908270.nnworker.queue;

import at.ac.univie.a00908270.nnworker.dl4j.Dl4jMnistNetworkTrainer;
import at.ac.univie.a00908270.nnworker.dl4j.Dl4jNetworkTrainer;
import at.ac.univie.a00908270.nnworker.util.NnStatus;
import at.ac.univie.a00908270.nnworker.util.Vinnsl;
import at.ac.univie.a00908270.vinnsl.schema.Parametervalue;
import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

import java.io.IOException;

@Component
public class Worker {
	
	private static final Logger log = LoggerFactory.getLogger(Worker.class);
	
	//	private static final String VINNSL_SERVICE_ENDPOINT = "http://127.0.0.1:8080";
	private static final String VINNSL_SERVICE_ENDPOINT = "http://vinnsl-service:8080";
	
	@Autowired
	WorkerQueue workerQueue;
	
	@Scheduled(fixedRate = 5000)
	public void reportCurrentTime() {
		
		if (!workerQueue.getQueue().isEmpty()) {
			
			String nnId = workerQueue.getQueue().poll();
			
			RestTemplate restTemplate = new RestTemplate();
			Vinnsl vinnslObject = restTemplate.getForObject(String.format(VINNSL_SERVICE_ENDPOINT + "/vinnsl/%s", nnId), Vinnsl.class);
			restTemplate.put(String.format(VINNSL_SERVICE_ENDPOINT + "/status/%s/%s", nnId, NnStatus.INPROGRESS), null);
			
			log.info("STARTING TRAINING OF " + nnId + vinnslObject);
			
			boolean isError = false;
			
			try {
				
				Parametervalue.Comboparameter specialWorkerClass = ((Parametervalue.Comboparameter) (vinnslObject.definition.getParameters().getValueparameterOrBoolparameterOrComboparameter().stream()
						.filter(e -> e instanceof Parametervalue.Comboparameter)
						.filter(e -> ((Parametervalue.Comboparameter) e).getName().equals("dl4jTrainerClass"))
						.findFirst().orElse(null)));
				
				if (specialWorkerClass != null && StringUtils.equalsIgnoreCase("at.ac.univie.a00908270.nnworker.dl4j.Dl4jMnistNetworkTrainer", specialWorkerClass.getValue())) {
					new Dl4jMnistNetworkTrainer(vinnslObject);
				} else {
					
					new Dl4jNetworkTrainer(vinnslObject);
				}
			} catch (IOException e) {
				isError = true;
				e.printStackTrace();
			} catch (InterruptedException e) {
				isError = true;
				e.printStackTrace();
			}
			/*
			try {
				Thread.sleep(10000);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}*/
			
			log.info("FINISHED TRAINING OF " + nnId + vinnslObject);
			
			if (!isError) {
				restTemplate.put(String.format(VINNSL_SERVICE_ENDPOINT + "/status/%s/%s", nnId, NnStatus.FINISHED), null);
			} else {
				restTemplate.put(String.format(VINNSL_SERVICE_ENDPOINT + "/status/%s/%s", nnId, NnStatus.ERROR), null);
			}
		} else {
			if (log.isDebugEnabled()) {
				log.debug("nothing to do");
			}
		}
	}
}