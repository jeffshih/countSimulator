import numpy as np 



class kalmanFilter(object):

    def __init__(self, Q = None, R = None, P = None, x0 = None):

    #transition Matrix
        self.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                                           [0, 1, 0, 0, 0, 1, 0],
                                           [0, 0, 1, 0, 0, 0, 1],
                                           [0, 0, 0, 1, 0, 0, 0],
                                           [0, 0, 0, 0, 1, 0, 0],
                                           [0, 0, 0, 0, 0, 1, 0],
                                           [0, 0, 0, 0, 0, 0, 1]])

        #outputMatrix
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]])

        #hard coded norm
        self.n = 7
        self.m = 7

        #process noise covariance 
        self.Q = np.array([[1, 0, 0, 0, 0, 0, 0],
                                           [0, 1, 0, 0, 0, 0, 0],
                                           [0, 0, 1, 0, 0, 0, 0],
                                           [0, 0, 0, 1, 0, 0, 0],
                                           [0, 0, 0, 0, 0.1, 0, 0],
                                           [0, 0, 0, 0, 0, 0.1, 0],
                                           [0, 0, 0, 0, 0, 0, 0.001]])

        #measurement noise matrix
        self.R = np.array([[1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 10, 0],
                                        [0, 0, 0, 10]])

        #estimate Error Matrix
        self.P = np.array([[10, 0, 0, 0, 0, 0, 0],
                                           [0, 10, 0, 0, 0, 0, 0],
                                           [0, 0, 10, 0, 0, 0, 0],
                                           [0, 0, 0, 10, 0, 0, 0],
                                           [0, 0, 0, 0, 1000, 0, 0],
                                           [0, 0, 0, 0, 0, 1000, 0],
                                           [0, 0, 0, 0, 0, 0, 1000]])

        self.x = np.zeros((self.n, 1)) if x0 is None else x0


    def update(self, z):

        #residual is the measured position - prediction position
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        #X = Xn+Ky
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        
        #P = (I-KH)Pn(I-KH)^T+KRKT

        #I-KH
        temp1 = np.dot(I-np.dot(K,self.H))
        #(I-KH)Pn
        temp2 = np.dot(temp1, self.p)
        
        #(I-KH)Transpose
        temp3 = (I-np.dot(K, self.H)).T

        #KRKT
        temp4 = np.dot(np.dot(K, self.R), K.T)
        res = np.dot(temp2,temp3)+temp4

        self.P = res
        '''
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
        	(I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)
        '''

        
    def predict(self):
        #remove B and U, no control input
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x 



if __name__ == "__main__":
    input = []
    kf = kalmanFilter()
    
