import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread


img = imread('meo.jpg')
img_tinted = img * [1, 0.95, 0.9]


plt.subplot(1, 2, 1)
plt.imshow(img)


plt.subplot(1, 2, 2)


plt.imshow(np.uint8(img_tinted))
plt.show()