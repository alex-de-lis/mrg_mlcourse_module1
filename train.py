import sys
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import argparse
from mnist import MNIST

class train():
    def load_X(self,dir):
        mndata = MNIST(dir)
        images = mndata.load_training()[0]
        self.images = np.array(images)

    def load_y(self,dir):
        mndata = MNIST(dir)
        labels = mndata.load_training()[1]
        self.labels = np.array(labels)

    def normalized(self,X):
        return X/255

    def add_bias(self,X):
        record_num, features_num = X.shape
        bias_f = np.ones((record_num, 1))
        return np.hstack((bias_f, X))

    def myresize(self,images):
        new_images=[]
        for img in images:
            def_img = np.reshape(img, (28, 28))
            new_img=[]
            for row in range(0,28,4):
                for col in range(0,28,4):
                    myslice = def_img[row:row+4,col:col+4]
                    str_arr = np.reshape(myslice, (16))
                    new_img.append(np.mean(str_arr))
            new_images.append(new_img)
        return np.array(new_images)

    def preprocessingX (self,X):
        resizeX=np.array(self.myresize(X))
        resizeX=self.normalized(resizeX)
        resizeX=self.add_bias(resizeX)
        return resizeX


    def dot_product(self,features, weights):
        inner_sum = 0
        inner_sum = np.dot(features, weights)
        return inner_sum


    def sign(self,x):
        return 1 if x >= 0.0 else -1


    def svm_predict(self,features, weights):
        score = np.dot(features, weights)
        pred_label = self.sign(score)
        return pred_label, score


    def init_weights(self,features_num, a, b):
        return a + (b - a) * np.random.random(features_num)

    def my_hinge_loss(self,predicted_value, true_value):
        result = max(0.0, 1.0 - predicted_value * true_value)
        return result

    def hinge_loss_dataset(self,predicted_values, true_values):
        a = 1 - true_values * predicted_values
        a[a < 0] = 0
        return np.sum(a) / a.shape[0]

    def hinge_loss_in_point(self,X, y, weights):
        predicted_values = list(map(lambda x: np.dot(x, weights), X))
        return self.hinge_loss_dataset(predicted_values, y)

    def num_gradient(self,loss_fun, X, y, model_weights, w_delta=0.01):
        current_loss = loss_fun(X, y, model_weights)
        weights_delta = model_weights[:]
        grad = []
        for coord in range(len(model_weights)):
            weights_delta[coord] += w_delta
            delta_loss = loss_fun(X, y, weights_delta)
            deriv = (delta_loss - current_loss) / w_delta
            grad.append(deriv)
            weights_delta[coord] -= w_delta
        return np.array(grad)

    def gradient_descent(self,loss_fun, X, y, initial_weights, learning_rate, iter_num, verbose=True):
        model_weights = initial_weights[:]
        for counter in range(iter_num):
            grad = self.num_gradient(loss_fun, X, y, model_weights)
            model_weights -= learning_rate * grad
            loss = loss_fun(X, y, model_weights)
            if verbose:
                print("Iter: %i, loss_value: %f" % (counter, loss))
        return model_weights

    def FindIndexes(self,y):
        mas=[]
        for digit in range(0,10):
            pos_ind = np.where(y == digit)
            neg_ind = np.where(y!= digit)
            mas.append([pos_ind, neg_ind])
        return mas

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

    def start(self,x_train_dir,y_train_dir,model_output_dir):
        self.load_X(x_train_dir)
        self.load_y(y_train_dir)
        print('load_complete')
        X = self.preprocessingX(self.images)
        print('preproc_complete')
        self.optimal_mas=[]
        X_train, X_test, y_train, y_test = train_test_split(X, self.labels, test_size=0.2, random_state=1)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
        index = self.FindIndexes(y_train)
        for i in range(0, 10):
            print("Start for digit: ", i)
            pos_ind, neg_ind = index[i]
            y_train[pos_ind] = 1
            y_train[neg_ind] = -1
            y_train.dtype = 'int8'
            initial_weights = self.init_weights(X_train.shape[1], -0.5, 0.5)
            learning_rate = 1
            iter_num = 100
            optimal_weights = self.gradient_descent(self.hinge_loss_in_point, X_train, y_train, initial_weights, learning_rate,iter_num)
            self.optimal_mas.append(optimal_weights)
            print("End for digit: ", i)
            print("Optimal weights: ", optimal_weights)
        self.optimal_mas=np.array(self.optimal_mas)
        np.savetxt(model_output_dir+'\\weights.txt', self.optimal_mas)
        self.ClasRep(X_test,y_test)

    def createParser(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('-x_train_dir', help='Train set')
        parser.add_argument('-y_train_dir', help='Train labels')
        parser.add_argument('-model_output_dir', help='Model output')

        return parser


if __name__ == "__main__":
    t=train()
    parser = t.createParser()
    namespace = parser.parse_args(sys.argv[1:])
    x_train_dir = namespace.x_train_dir
    y_train_dir = namespace.y_train_dir
    model_output_dir=namespace.model_output_dir
    t.start(x_train_dir,y_train_dir,model_output_dir)