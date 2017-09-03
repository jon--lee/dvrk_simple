import numpy as np
import keras
import learner
import knet
from sklearn.svm import OneClassSVM
import cv2
import pickle
import IPython

class DataExtractor():
    """
        Extracts delta poses from trajectory files
        and gets images for provided directories. Must
        specify length of trajectory.
    """


    def __init__(self, direcs, T):
        self.direcs = direcs
        self.T = T
        self.trajs = None
        self.X = None
        self.y = None


    @staticmethod
    def get_pos(traj_t):
        pos = np.zeros(6)
        pos_raw = traj_t['pos']
        rot_raw = traj_t['rot']
        pos[0] = pos_raw[0][0,0]
        pos[1] = pos_raw[1][0,0]
        pos[2] = pos_raw[2][0,0]
        pos[3] = rot_raw[0]
        pos[4] = rot_raw[1]
        pos[5] = rot_raw[2]
        return pos 


    def extract(self):
        trajs = []
        X = []
        y = []
        for direc in self.direcs:
            f = open('demos/' + direc + '/traj.p', 'r')
            traj = []
            poses = []
            for t in range(self.T):
                traj_t = pickle.load(f)
                traj.append(traj_t)
                poses.append(self.get_pos(traj_t))

            states = []
            controls = []
            for t in range(self.T - 1):
                im = cv2.imread('demos/' + direc + '/left' + str(t) + '.jpg')
                small_im = cv2.resize(im, (100, 100)) / 255.0           # can adjust this
                pose = poses[t]
                pose_next = poses[t + 1]

                u = pose_next - pose
                controls.append(u)
                states.append(small_im.flatten())

            self.poses = poses
            X += states
            y += controls
            trajs.append(states)

        self.trajs = trajs
        self.X, self.y = X, y
        return trajs, X, y



def main():
    direcs = ['OUV9DC3LZ08N6R6DJYQF']
    de = DataExtractor(direcs, 137)
    trajs, X, y = de.extract()

    est = knet.Network([64,64], learning_rate = .01, epochs = 50)
    lnr = learner.Learner(est)
    lnr.add_data(X, y)
    lnr.train()

    print "Control at init state: " + str(lnr.intended_action(X[0]))

    lnr.est.save_weights('tmp/weights.txt', 'tmp/stats.txt')
    lnr.est.load_weights('tmp/weights.txt', 'tmp/stats.txt')

    print "Control at init stat after loading: " + str(lnr.intended_action(X[0]))

    




if __name__ == '__main__':
    main()


