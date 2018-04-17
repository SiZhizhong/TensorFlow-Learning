import  tensorflow as  tf
import  numpy as  np
import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



###construst graph

class LinearModel:
    def __init__(self,input_feature_size,output_feature_size):
        self.input_feature_size=input_feature_size
        self.output_feature_size=output_feature_size
        self.trained=False
        self.cofigure_model()


    def cofigure_model(self):
        self.X = tf.placeholder(tf.float32, [self.input_feature_size, None], name="X")
        self.Y = tf.placeholder(tf.float32, [self.output_feature_size, None], name="Y")

        self.W = tf.get_variable("Weight", [int(self.input_feature_size/2), self.input_feature_size], tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.1))
        self.bias = tf.get_variable("bias", [int(self.input_feature_size/2),1], tf.float32,
                                    initializer=tf.zeros_initializer)
        self.logits=tf.matmul(self.W,self.X)+self.bias
        self.activation=tf.nn.tanh(self.logits)

        self.W1=tf.get_variable("Weight2", [self.output_feature_size, int(self.input_feature_size/2)], tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.1))
        self.bias1 = tf.get_variable("bias2", [self.output_feature_size,1], tf.float32,
                                    initializer=tf.zeros_initializer)
        self.output_sigmoid=tf.matmul(self.W1,self.activation)+self.bias1
        self.prediction=tf.nn.sigmoid(self.output_sigmoid)
        self.loss=tf.reduce_mean(tf.square(self.prediction-self.Y))



    def train(self,print_eopoch=False):

        self.optimizer=tf.train.AdadeltaOptimizer(0.0001).minimize(self.loss)
        self.saver=tf.train.Saver(max_to_keep=5)
        for i in range(1001):
            with tf.Session() as sess:
                sess.writer=tf.summary.FileWriter("./graphs",sess.graph)
                sess.run(tf.global_variables_initializer())
                _,l=sess.run([self.optimizer,self.loss],feed_dict={self.X:self.data_x,self.Y:self.data_y})
                if print_eopoch and i %50==0:
                    print("Epoch ",i,": ","loss: ",l)
                    self.saver.save(sess,"./checkpoint/LinearModel")
        self.trained=True


    def load_data(self,path):
        file=open(path,'r')
        data=np.asarray([row for row in csv.reader(file)])
        data=data[1:,0:]
        for i in range(len(data)):
            if data[i][4]=="Present":
                data[i][4]=1
            else:
                data[i][4]=0
        self.data_x=data[:,0:9].T
        self.data_y=data[:,9].reshape(1,-1)


if __name__=="__main__":
    mymodel=LinearModel(9,1)
    mymodel.load_data("./data/heart.csv")
    mymodel.train(True)