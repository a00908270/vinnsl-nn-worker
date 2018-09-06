package at.ac.univie.a00908270.nnworker.queue;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.LinkedList;
import java.util.Queue;

@Configuration
public class WorkerQueue {
	Queue<String> queue = new LinkedList<String>();
	
	@Bean
	public Queue<String> getQueue() {
		return queue;
	}
}
