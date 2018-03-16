import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.mapstruct.Mapper;
import org.mapstruct.Mapping;
import org.mapstruct.Mappings;
import org.mapstruct.factory.Mappers;

@Mapper
public interface VinnslDL4JMapper {
	VinnslDL4JMapper INSTANCE = Mappers.getMapper(VinnslDL4JMapper.class);
	
	/*	@Mapping(source = "activationfunction", target = "activation")
		NeuralNetConfiguration.Builder neuralNetConfiguration(Instanceschema instance);*/
	@Mappings({
			@Mapping(source = "vinnsl.trainingresult.learningrate", target = "learningRate"),
			@Mapping(source = "vinnsl.trainingresult.epochs", target = "numIterations"),
			//@Mapping(source = "vinnsl.instance.activationfunction", target = "activationFn"),
			@Mapping(source = "vinnsl.instance.weightmatrix", target = "weightInit")
	})
	NeuralNetConfiguration.Builder neuralNetConfiguration(Vinnsl vinnsl);
}
