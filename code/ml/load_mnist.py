import matplotlib.pyplot as plt
from keras.datasets import mnist 
# mnist dataset is included in Keras datasets library
(X_train, y_train), (X_test, y_test) = mnist.load_data() 

# size of training data (the 'shape' of the array)
print("Training data size:")
print(X_train.shape)
#print(X_train.shape[0])

print("Test data size:")
print(X_test.shape)

# display and example image 
sample = 0 # put the image number you want to see here (0 - 59999)
plt.title('Label is {label}'.format(label=y_train[sample]))
plt.imshow(X_train[sample], cmap='gray')
plt.show()


