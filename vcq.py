import numpy as np
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit_algorithms.utils import algorithm_globals

algorithm_globals.random_seed = 3142
np.random.seed(algorithm_globals.random_seed)
FEATURE_MAP = ZZFeatureMap(feature_dimension=2, reps=2)
VAR_FORM = TwoLocal(2, ["ry", "rz"], "cz", reps=2)

AD_HOC_CIRCUIT = FEATURE_MAP.compose(VAR_FORM)
AD_HOC_CIRCUIT.measure_all()
AD_HOC_CIRCUIT.decompose().draw()
print(AD_HOC_CIRCUIT)
