package at.ac.univie.a00908270.nnworker.vinnsl.transformation;

import at.ac.univie.a00908270.nnworker.util.Vinnsl;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.mapstruct.Mapper;
import org.mapstruct.Mapping;
import org.mapstruct.Mappings;
import org.mapstruct.factory.Mappers;

@Mapper(uses = MappingUtil.class)
public interface VinnslDL4JMapper {
	VinnslDL4JMapper INSTANCE = Mappers.getMapper(VinnslDL4JMapper.class);
	
	/*@Mapping(source = "activationfunction", target = "activation")
	NeuralNetConfiguration.Builder neuralNetConfiguration(Instanceschema instance);*/
	
	/*@Mappings({
			@Mapping(source = "vinnsl.trainingresult.learningrate", target = "learningRate"),
			@Mapping(source = "vinnsl.trainingresult.epochs", target = "numIterations"),
			//@Mapping(source = "vinnsl.instance.activationfunction", target = "activationFn"),
			@Mapping(source = "vinnsl.instance.weightmatrix", target = "weightInit")
	})*/
	/*NeuralNetConfiguration.Builder neuralNetConfiguration(Vinnsl vinnsl);*/
	
	@Mappings({
			// learningrate -> learningRate
			// momentum -> momentum
			// biasInput -> biasInit
			// epochs -> numIterations
			// threshold -> gradientNormalizationThreshold
			// activationfunction -> (ENUM) activationFn
			@Mapping(source = "vinnsl.definition.parameters.valueparameterOrBoolparameterOrComboparameter", target = "learningRate", qualifiedBy = MappingUtil.LearningRate.class),
			@Mapping(source = "vinnsl.definition.parameters.valueparameterOrBoolparameterOrComboparameter", target = "momentum", qualifiedBy = MappingUtil.Momentum.class),
			@Mapping(source = "vinnsl.definition.parameters.valueparameterOrBoolparameterOrComboparameter", target = "biasInit", qualifiedBy = MappingUtil.BiasInit.class),
			@Mapping(source = "vinnsl.definition.parameters.valueparameterOrBoolparameterOrComboparameter", target = "numIterations", qualifiedBy = MappingUtil.NumIterations.class),
			@Mapping(source = "vinnsl.definition.parameters.valueparameterOrBoolparameterOrComboparameter", target = "gradientNormalizationThreshold", qualifiedBy = MappingUtil.GradientNormalizationThreshold.class),
			@Mapping(source = "vinnsl.definition.parameters.valueparameterOrBoolparameterOrComboparameter", target = "activationFn", qualifiedBy = MappingUtil.ActivationFn.class)
	})
	NeuralNetConfiguration.Builder neuralNetConfiguration(Vinnsl vinnsl);
	
	
}
