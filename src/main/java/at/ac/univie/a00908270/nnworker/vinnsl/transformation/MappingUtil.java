package at.ac.univie.a00908270.nnworker.vinnsl.transformation;

import at.ac.univie.a00908270.vinnsl.schema.Parametervalue;
import org.mapstruct.Qualifier;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.activations.impl.ActivationTanH;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

public class MappingUtil {
	
	// learningrate -> learningRate
	// momentum -> momentum
	// biasInput -> biasInit
	// iterations -> numIterations
	// threshold -> gradientNormalizationThreshold
	// activationfunction -> (ENUM) activationFn
	
	
	@Qualifier
	@Target(ElementType.METHOD)
	@Retention(RetentionPolicy.SOURCE)
	public @interface LearningRate {
	}
	
	@Qualifier
	@Target(ElementType.METHOD)
	@Retention(RetentionPolicy.SOURCE)
	public static @interface Momentum {
	}
	
	@Qualifier
	@Target(ElementType.METHOD)
	@Retention(RetentionPolicy.SOURCE)
	public static @interface BiasInit {
	}
	
	@Qualifier
	@Target(ElementType.METHOD)
	@Retention(RetentionPolicy.SOURCE)
	public static @interface NumIterations {
	}
	
	@Qualifier
	@Target(ElementType.METHOD)
	@Retention(RetentionPolicy.SOURCE)
	public static @interface GradientNormalizationThreshold {
	}
	
	@Qualifier
	@Target(ElementType.METHOD)
	@Retention(RetentionPolicy.SOURCE)
	public static @interface ActivationFn {
	}
	
	@LearningRate
	public Double learningRate(java.util.List<java.lang.Object> in) {
		Parametervalue.Valueparameter param = ((Parametervalue.Valueparameter) (in.stream()
				.filter(e -> e instanceof Parametervalue.Valueparameter)
				.filter(e -> ((Parametervalue.Valueparameter) e).getName().equals("learningrate"))
				.findFirst().orElse(null)));
		
		if (param != null) {
			return param.getValue().doubleValue();
		}
		
		return 0.1;
	}
	
	@Momentum
	public Double momentum(java.util.List<java.lang.Object> in) {
		Parametervalue.Valueparameter param = ((Parametervalue.Valueparameter) (in.stream()
				.filter(e -> e instanceof Parametervalue.Valueparameter)
				.filter(e -> ((Parametervalue.Valueparameter) e).getName().equals("momentum"))
				.findFirst().orElse(null)));
		
		if (param != null) {
			return param.getValue().doubleValue();
		}
		
		return 0d;
	}
	
	@BiasInit
	public Double biasInit(java.util.List<java.lang.Object> in) {
		
		Parametervalue.Valueparameter param = ((Parametervalue.Valueparameter) (in.stream()
				.filter(e -> e instanceof Parametervalue.Valueparameter)
				.filter(e -> ((Parametervalue.Valueparameter) e).getName().equals("biasInput"))
				.findFirst().orElse(null)));
		
		if (param != null) {
			return param.getValue().doubleValue();
		}
		
		return 0d;
	}
	
	@NumIterations
	public Double numIterations(java.util.List<java.lang.Object> in) {
		
		Parametervalue.Valueparameter param = ((Parametervalue.Valueparameter) (in.stream()
				.filter(e -> e instanceof Parametervalue.Valueparameter)
				.filter(e -> ((Parametervalue.Valueparameter) e).getName().equals("iterations"))
				.findFirst().orElse(null)));
		
		if (param != null) {
			return param.getValue().doubleValue();
		}
		
		return 0d;
	}
	
	@GradientNormalizationThreshold
	public Double gradientNormalizationThreshold(java.util.List<java.lang.Object> in) {
		
		Parametervalue.Valueparameter param = ((Parametervalue.Valueparameter) (in.stream()
				.filter(e -> e instanceof Parametervalue.Valueparameter)
				.filter(e -> ((Parametervalue.Valueparameter) e).getName().equals("threshold"))
				.findFirst().orElse(null)));
		
		if (param != null) {
			return param.getValue().doubleValue();
		}
		
		return 0d;
	}
	
	@ActivationFn
	public IActivation activationFn(java.util.List<java.lang.Object> in) {
		
		Parametervalue.Comboparameter param = ((Parametervalue.Comboparameter) (in.stream()
				.filter(e -> e instanceof Parametervalue.Comboparameter)
				.filter(e -> ((Parametervalue.Comboparameter) e).getName().equals("activationfunction"))
				.findFirst().orElse(null)));
		
		if (param != null) {
			if ("sigmoid".equalsIgnoreCase(param.getValue())) {
				return new ActivationSigmoid();
			}
			if ("tanh".equalsIgnoreCase(param.getValue())) {
				return new ActivationTanH();
			}
		}
		
		return null;
	}
}
