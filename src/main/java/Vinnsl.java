import at.ac.univie.a00908270.vinnsl.schema.*;

import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlRootElement;

@XmlRootElement
public class Vinnsl {
	
	public String identifier;
	
	@XmlElement
	public Description description;
	
	@XmlElement
	public Definition definition;
	
	@XmlElement
	public Dataschema data;
	
	@XmlElement
	public Instanceschema instance;
	
	@XmlElement
	public Trainingresultschema trainingresult;
	
	@XmlElement
	public Resultschema result;
	
	
	public Vinnsl() {
	}
	
	
	@Override
	public String toString() {
		return String.format(
				"VinnslDefinition[identifier=%s, definition='%s']",
				identifier, definition);
	}
	
}
