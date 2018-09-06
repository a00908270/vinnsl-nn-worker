package at.ac.univie.a00908270.nnworker.util;

public class NnCloud {
	protected NnStatus status;
	
	protected String dl4jNetwork;
	
	public NnCloud() {
		this.status = NnStatus.CREATED;
	}
	
	public NnStatus getStatus() {
		return status;
	}
	
	public void setStatus(NnStatus status) {
		this.status = status;
	}
}
