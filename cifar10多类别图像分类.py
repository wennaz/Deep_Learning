#CIFAR10 数据集包含 10 类，共 60000 张彩色图片，每类图片有 6000 张。此数据集中 50000 个样例被作为训练集，剩余 10000 个样例作为测试集。类之间相互度立，不存在重叠的部分。因此，cifar10分类就是一个图像多分类任务。Keras另一个好处在于已经集成了很多常见的数据集和模型，在接口里可以直接调用。
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
#CIFAR10 数据集包含 10 类，共 60000 张彩色图片，每类图片有 6000 张。此数据集中 50000 个样例被作为训练集，剩余 10000 个样例作为测试集。类之间相互度立，不存在重叠的部分。
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# 将像素的值标准化至0到1的区间内。
train_images, test_images = train_images / 255.0, test_images / 255.0
#我们将测试集的前 25 张图片和类名打印出来，来确保数据集被正确加载。
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # 由于 CIFAR 的标签是 array， 
    # 因此您需要额外的索引（index）。
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
#下方展示的 6 行代码声明了了一个常见卷积神经网络，由几个 Conv2D 和 MaxPooling2D 层组成。
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#在模型的最后，您将把卷积后的输出张量（本例中形状为 (4, 4, 64)）传给一个或多个 Dense 层来完成分类。
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.summary()

#编译并训练模型
model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

#评估模型
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)
