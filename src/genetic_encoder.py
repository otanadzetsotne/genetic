from src.genetic import Genetic, ModelGenerator
from src.dtypes import Individual


class _ModelGeneratorEncoder(ModelGenerator):
    @classmethod
    def _generate_output(
            cls,
            individual: Individual,
            input_shape: int,
            output_shape: int,
            structure,
    ):
        structure = cls._add_layers([[output_shape, 'relu']], structure)
        structure = cls._add_layers(individual[::-1], structure)
        structure = cls._add_layers([[input_shape, 'relu']], structure)

        return structure


class GeneticEncoder(Genetic):
    @property
    def model_generator(self) -> ModelGenerator:
        return _ModelGeneratorEncoder()
