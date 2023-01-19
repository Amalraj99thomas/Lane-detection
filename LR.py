import numpy as np
import matplotlib.pyplot as plt

def draw(x1, x2):
    line =plt.plot(x1, x2, '-')
    plt.pause(0.001)
    line[0].remove()

def sigmoid(score):
    return 1/(1 + np.exp(-score))


def calculate_error(line_parameters, points , y):
  n = points.shape[0]
  p = sigmoid(points*line_parameters)
  cross_entropy =-(1/n)*(np.log(p).T * y + np.log(1-p).T * (1-y))
  return cross_entropy

#alpha is the learning rate.
def gradient_descent(line_parameters, points, y, alpha):
    m = points.shape[0]
    for i in range(2000):
        p = sigmoid(points*line_parameters)
        gradient = (points.T * (p-y)) * (alpha / m)
        line_parameters = line_parameters - gradient
        w1 = line_parameters.item(0)
        w2 = line_parameters.item(1)
        b = line_parameters.item(2)
        x1 = np.array([points[:, 0].min(), points[:, 0].max()])
        x2 = -b / w2 + x1 * (-w1 / w2)
        draw(x1, x2)

#for top region of data points
n_pts = 100
np.random.seed(0) #to make the same random values to appear
bias = np.ones(n_pts)
#random_x1_values =
##assigns randomly distributed values for horizontal coordinates
#random_x2_values =
##assigns randomly distributed values for vertical coordinates
top_region = np.array([np.random.normal(10, 2, n_pts) , np.random.normal(12, 2, n_pts), bias ]).T
#similarly for bottom region of data
bottom_region = np.array([np.random.normal(5, 2, n_pts) , np.random.normal(6, 2, n_pts), bias ]).T
#creating the line to classify
all_points = np.vstack((top_region, bottom_region))
# w1 = -0.2
# w2 = -0.35
# b = 3.5
#line_parameters = np.matrix([w1, w2, b]).T #weights and bias #Transpose to maintain compatibility
line_parameters = np.matrix(np.zeros(3)).T
#print(all_points.shape)
#print(line_parameters.shape)
#print(x1, x2)
# linear_combo = all_points * line_parameters
# probabilities = sigmoid(linear_combo)
# probabilities
y = np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts*2, 1) #label of all points
#top red is all labelled 0, and bottom blue is labelled as 1


_, ax = plt.subplots(figsize = (4, 4))
ax.scatter(top_region[:, 0], top_region[:, 1], color ='r')
ax.scatter(bottom_region[:, 0], bottom_region[:, 1], color ='b')
# draw(x1, x2)
gradient_descent(line_parameters, all_points, y, 0.06)
plt.show()

print((calculate_error(line_parameters, all_points, y)))
