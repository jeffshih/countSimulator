import numpy as np 



class kalmanFilter(object):

    def __init__(self, F = None, H = None, Q = None, R = None, P = None, x0 = None):

        
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
        self.m = 4

        self.x = np.zeros((self.n, 1))
        self.x[0][0] = x0[0][0]
        self.x[1][0] = x0[1][0]
        self.x[2][0] = x0[2][0]
        self.x[3][0] = x0[3][0]
        self.B = np.ones((self.m, 1))

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
                                        [0, 0, 3, 0],
                                        [0, 0, 0, 3]])

        #estimate Error Matrix
        self.P = np.array([[10, 0, 0, 0, 0, 0, 0],
                                           [0, 10, 0, 0, 0, 0, 0],
                                           [0, 0, 10, 0, 0, 0, 0],
                                           [0, 0, 0, 10, 0, 0, 0],
                                           [0, 0, 0, 0, 1000, 0, 0],
                                           [0, 0, 0, 0, 0, 1000, 0],
                                           [0, 0, 0, 0, 0, 0, 1000]])



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
        temp1 = I-np.dot(K,self.H)
        #(I-KH)Pn
        temp2 = np.dot(temp1, self.P)
        
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

        
    def predict(self, u=0):
        #use control input, we have strong assumption that object is moving left
        self.x = np.dot(self.F, self.x)#+np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x 



if __name__ == "__main__":
    input = []
    kf = kalmanFilter()
    
