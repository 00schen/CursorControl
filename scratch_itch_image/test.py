import numpy as np
from numpy.linalg import norm
import os

from utils import Predictor

dirname = os.path.dirname(__file__)
predictor_path = os.path.join(dirname,'demo','final_model.h5')
data_path = os.path.join(dirname,'samples','ScratchItchJaco-v0','noised_action.npz')

X,y = np.load(data_path).values()

predictor = Predictor(predictor_path)

i = 0
for data,target in zip(X,y):
    if i > 5:
        break
    i += 1
    predictor.reset()
    for step in data:
        print(step.shape)
        print(norm(predictor.predict(step)-target))