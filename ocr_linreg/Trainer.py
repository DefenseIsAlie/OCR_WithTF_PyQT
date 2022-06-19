import os 
import numpy as cp

class Linreg:
    def TestX():
        X = cp.load("/home/abhishekj/Github/OCR_WithTF_PyQT/XY/X.npy")
        x = X.T
        print(type(X[0][1]))

    def DirectSolver():
        X = cp.load("/home/abhishekj/Github/OCR_WithTF_PyQT/XY/X.npy")
        Y = cp.load("/home/abhishekj/Github/OCR_WithTF_PyQT/XY/Y.npy")
        x = X.T

        tmp = cp.linalg.inv(cp.dot(x,X))
        tmp = cp.dot(tmp,x)
        W   = tmp @ Y

        cp.save("W",W)


    def Trainer():
        X = cp.load("/home/abhishekj/Github/OCR_WithTF_PyQT/XY/X.npy")
        Y = cp.load("/home/abhishekj/Github/OCR_WithTF_PyQT/XY/Y.npy")

        W = cp.zeros(X.shape[1])
        b = 0
        N = X.shape[0]

        iter  = 10
        alpha = 0.0000000001

        for i in range(iter):
            print(i)
            ypred = cp.dot(X,W)+b
            print(ypred[:10])

            dw    = (2/N)*alpha*cp.dot(X.T,(ypred-Y))
            db    = (2/N)*alpha*cp.sum(ypred-Y)

            W -= dw
            b -= db


            
        print(b)
        print(type(b))

    def PolyTrainerSGD():
        y = cp.load("/home/abhishekj/Github/OCR_WithTF_PyQT/XY/Y.npy")

        W1 = cp.load("/home/abhishekj/Github/OCR_WithTF_PyQT/W.npy")
        W2 = cp.zeros(W1.shape)
        W3 = cp.copy(W1.shape)

        b = 0
        N = y.shape[0]


        iter = 10000
        alpha = 0.00000001

        for i in range(iter):
            X = cp.load("/home/abhishekj/Github/OCR_WithTF_PyQT/XY/"+"x"+str(i%10)+".npy")
            t = i%10
            Y = y[int(t*N/10):int((t+1)*N/10)]

            ypred = cp.dot(X,W1) + cp.dot(X,W2) + cp.dot(X,W3) + b