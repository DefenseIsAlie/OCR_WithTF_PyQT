import os 
from PIL import Image
import numpy as cp

class dataprocessor:

    def process():
        os.chdir("/home/abhishekj/Github/OCR_WithTF_PyQT/renamedData")

        pth = os.getcwd()
        outfile = "/home/abhishekj/Github/OCR_WithTF_PyQT/"+"XY"

        Langs = os.listdir()

        cls = []
        L   = []

        for lang in Langs:
            folders = os.listdir(pth+"/"+lang)
            #x = []
            #y = []
            for folder in folders:
                c = int(folder)
                imgs = os.listdir(pth+"/"+lang+"/"+folder)
                if lang == "Telugu" and len(imgs) > 100:
                    imgs = imgs[:100]
                for img in imgs:
                    im = Image.open(pth+"/"+lang+"/"+folder+"/"+img).convert("1")
                    i  = cp.asarray(im,dtype="int").flatten()
                    im.close()
                    #y.append(c)
                    #x.append(i)
                    cls.append(c)
                    L.append(i)
            #x = cp.vstack(x)
            #y = cp.asarray(y)
            #cp.save(outfile+"/"+lang+"X",x)
            #cp.save(outfile+"/"+lang+"Y",y)
        
        print(L[0].shape)
            

        X   = cp.vstack(L)
        Y   = cp.asarray(cls)
        print(X[0][:10])
        print(X.shape)

        cp.save(outfile+"/"+"X",X)
        cp.save(outfile+"/"+"Y",Y)

    def splitX(pth):
        X = cp.load(pth)
        N = X.shape[0]
        for i in range(100):
            x = X[int(i*N/100):int((i+1)*N/100),:]
            cp.save("/home/abhishekj/Github/OCR_WithTF_PyQT/XY/"+"x"+str(i),x)
