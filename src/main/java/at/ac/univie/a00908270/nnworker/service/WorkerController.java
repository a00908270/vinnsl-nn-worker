package at.ac.univie.a00908270.nnworker.service;

import at.ac.univie.a00908270.nnworker.queue.WorkerQueue;
import at.ac.univie.a00908270.nnworker.util.NnStatus;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;

import java.util.Queue;


@RestController
@RequestMapping("/worker")
public class WorkerController {
	
	@Autowired
	WorkerQueue workerQueue;
	
	@GetMapping(value = "/queue")
	public ResponseEntity<Queue<String>> getWorkingQueue() {
		return ResponseEntity.ok().body(workerQueue.getQueue());
	}
	
	@PutMapping(value = "/queue/{id}", produces = MediaType.APPLICATION_JSON_VALUE)
	public ResponseEntity<Queue<String>> addToWorkingQueue(@PathVariable("id") String id) {
		workerQueue.getQueue().add(id);
		
		RestTemplate restTemplate = new RestTemplate();
		restTemplate.put(String.format("http://127.0.0.1:8080/status/%s/%s", id, NnStatus.QUEUED), null);
		
		return ResponseEntity.ok().body(workerQueue.getQueue());
	}
	
	
}
