from lib2to3.pytree import convert
import numpy as cp
from PIL import Image



class Predictor:
    p = ""
    def __init__(self,path) -> None:
        self.imgpath = path
        p = path 

    def predict(self):
        X = Image.open(self.imgpath).convert("1")
        X = cp.asarray(X,dtype="int").flatten()
        W = cp.load("/home/abhishekj/Github/OCR_WithTF_PyQT/W.npy")
        pred = cp.dot(W,X)
        return pred