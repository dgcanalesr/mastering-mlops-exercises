import bentoml
import numpy as np

from bentoml.io import JSON, NumpyNdarray
from pydantic import BaseModel


iris_qda_clf_runner = bentoml.sklearn.get("iris_qda_clf:latest").to_runner()

clf = bentoml.Service("iris_classifier", runners=[iris_qda_clf_runner])


class IrisFeatures(BaseModel):
    sepal_len: float
    sepal_width: float
    petal_len: float
    petal_width: float


@clf.api(
    input=JSON(pydantic_model=IrisFeatures),
    output=NumpyNdarray()
)
async def classify(input: IrisFeatures) -> np.ndarray:
    input_array = np.array([
        [input.sepal_len,
        input.sepal_width,
        input.petal_len,
        input.petal_width]
    ])
    return await iris_qda_clf_runner.predict.async_run(input_array)