import numpy as np

def lbp(img):
    l = []
    k = [[1,2,4],
         [8,0,16],
         [32,64,128]]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if i < 1 or i >= img.shape[0]-1 or j < 1 or j >= img.shape[1]-1:
                l.append(0)
            else:
                theshold = img[i][j]
                value = 0
                if img[i-1][j-1] >= theshold:
                    value += k[0][0]
                if img[i-1][j] >= theshold:
                    value += k[0][1]
                if img[i-1][j+1] >= theshold:
                    value += k[0][2]
                if img[i][j-1] >= theshold:
                    value += k[1][0]
                if img[i][j+1] >= theshold:
                    value += k[1][2]
                if img[i+1][j-1] >= theshold:
                    value += k[2][0]
                if img[i+1][j] >= theshold:
                    value += k[2][1]
                if img[i+1][j+1] >= theshold:
                    value += k[2][2]
                l.append(value)
    l = np.array(l)
    return l

data = np.load('./new_enhance2_7data.npy')
data_2d = []
for d in data:
    data_2d.append(d.reshape((48,48)))
data_2d = np.array(data_2d)

data_lbp = []
for d in data_2d:
    data_lbp.append(lbp(d))
data_lbp = np.array(data_lbp)

np.save('./lbp_7data.npy', data_lbp)