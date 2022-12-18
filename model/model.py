from pydantic import BaseModel
from typing import List, Optional

class Wine(BaseModel):
    id: Optional[int]
    fixed_acidity: float
    volatile_acidity: float
    citric_acidity: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    ph: float
    sulphates: float
    alcohol: float

class Parameters(BaseModel):
    C: float
    gamma: float
    kernel: str
    

class Metrics(BaseModel):
    break_ties: str
    cache_size: str
    class_weight: Optional[str]
    coef0: int
    decision_function_shape: str
    degree: int
    max_iter: int
    probability: bool
    random_state: Optional[int]
    shrinking: bool
    tol: float
    verbose: bool


class Model(BaseModel):
    parameters: Parameters
    metrics: Metrics