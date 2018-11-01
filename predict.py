import sys
import numpy as np
from sklearn.metrics import classification_report
from scipy.misc import imresize
from mnist import MNIST
import argparse

class predict():
    def load_X(self,dir):
        mndata = MNIST(dir)
        images = mndata.load_testing()[0]
        self.images = np.array(images)

    def load_y(self,dir):
        mndata = MNIST(dir)
        labels = mndata.load_testing()[1]
        self.labels = np.array(labels)

    def normalized(self, X):
        return X / 255

    def add_bias(self, X):
        record_num, features_num = X.shape
        bias_f = np.ones((record_num, 1))
        return np.hstack((bias_f, X))

    def myresize(self, images):
        new_images = []
        for img in images:
            def_img = np.reshape(img, (28, 28))
            new_img = []
            for row in range(0, 28, 4):
                for col in range(0, 28, 4):
                    myslice = def_img[row:row + 4, col:col + 4]
                    str_arr = np.reshape(myslice, (16))
                    new_img.append(np.mean(str_arr))
            new_images.append(new_img)
        return np.array(new_images)

    def preprocessingX(self, X):
        resizeX = np.array(self.myresize(X))
        resizeX = np.array(resizeX)
        resizeX = self.normalized(resizeX)
        resizeX = self.add_bias(resizeX)
        return resizeX

    def dot_product(self, features, weights):
        inner_sum = np.dot(features, weights)
        return inner_sum

    def sign(self, x):
        return 1 if x >= 0.0 else -1

    def svm_predict(self, features, weights):
        score = np.dot(features, weights)
        pred_label = self.sign(score)
        return pred_label, score

    def multNatr(self,x):
        s=0
        for i in range(0,10):
            s+=np.exp(np.dot(self.optimal_mas[i],x))
        return s

    def softmax(self,x):
        result=0
        maxP=0
        s=self.multNatr(x)
        for i in range(0,10):
            temp=np.exp(np.dot(self.optimal_mas[i],x))/s
            if(maxP<temp):
                result=i
                maxP=temp
        return result, maxP

    def ClasRep(self,X_test,y_test):
        predicted_labels = []
        for i, record in enumerate(X_test):
            predicted_labels.append(self.softmax(record)[0])
        cl = classification_report(y_test, predicted_labels)
        print(cl)

    def load_model(self,model_input_dir):
        self.optimal_mas=np.loadtxt(model_input_dir+'\\weights.txt')


    def start(self,x_train_dir,y_train_dir,model_input_dir):
        self.load_X(x_train_dir)
        self.load_y(y_train_dir)
        self.load_model(model_input_dir)
        print('load_complete')
        X_test = self.preprocessingX(self.images)
        y_test=self.labels
        print('preproc_complete')
        self.ClasRep(X_test,y_test)

    def createParser(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('-x_train_dir', help='Train set')
        parser.add_argument('-y_train_dir', help='Train labels')
        parser.add_argument('-model_input_dir', help='Model input')

        return parser

if __name__ == "__main__":
    p=predict()
    parser = p.createParser()
    namespace = parser.parse_args(sys.argv[1:])
    x_train_dir=namespace.x_train_dir
    y_train_dir=namespace.y_train_dir
    model_input_dir=namespace.model_input_dir
    p.start(x_train_dir,y_train_dir,model_input_dir)