import numpy
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error, r2_score
from pykalman import KalmanFilter

observations = numpy.loadtxt("observations_test.csv")
measurements = numpy.loadtxt("measurements_test.csv")

kf = KalmanFilter(initial_state_mean=observations[0])
kf = kf.em(observations, n_iter=5, em_vars='all')
measurements_predicted = (kf.smooth(measurements)[0])[:, 0]

trainScore1 = math.sqrt(mean_squared_error(observations, measurements))
trainScore1_r2 = r2_score(observations, measurements)
print('Score Before Kalman Filtering: %.2f RMSE and %.2f R-Square' % (trainScore1, trainScore1_r2))

trainScore2 = math.sqrt(mean_squared_error(observations, measurements_predicted))
trainScore2_r2 = r2_score(observations, measurements_predicted)
print('Score After Kalman Filtering: %.2f RMSE and %.2f R-Square' % (trainScore2, trainScore2_r2))

line1, = plt.plot(observations, label="Observations", linestyle='--')
line2, = plt.plot(measurements, label="Measurements", linestyle='--')
line3, = plt.plot(measurements_predicted, label="Improved Measurements", linestyle='--')

plt.legend(handles=[line1, line2, line3], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0., mode='expand',
           ncol=3)

#plt.xlabel('Time')
#plt.ylabel('Temperature')
plt.suptitle('Before: %.2f RMSE and %.2f R-Square | After: %.2f RMSE and %.2f R-Square' % (
trainScore1, trainScore1_r2, trainScore2, trainScore2_r2))

plt.show()
