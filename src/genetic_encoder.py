from src.genetic import Genetic, ModelGenerator
from src.dtypes import Individual


class _ModelGeneratorEncoder(ModelGenerator):
    @classmethod
    def _generate_output(
            cls,
            individual: Individual,
            output_shape: int,
            structure,
    ):
        # TODO
        pass


class GeneticEncoder(Genetic):
    @property
    def model_generator(self) -> ModelGenerator:
        return _ModelGeneratorEncoder()
