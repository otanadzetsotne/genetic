from src.genetic import Genetic, ModelGenerator
from src.dtypes import Individual


class _ModelGeneratorPerceptron(ModelGenerator):
    @classmethod
    def _generate_output(
            cls,
            individual: Individual,
            output_shape: int,
            structure,
    ):
        return cls._add_layers([[output_shape, 'relu']], structure)


class GeneticPerceptron(Genetic):
    @property
    def model_generator(self) -> ModelGenerator:
        return _ModelGeneratorPerceptron()
