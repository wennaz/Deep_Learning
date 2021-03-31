# 神经风格迁移是一种优化技术，用于将两个图像——一个内容图像和一个风格参考图像（如著名画家的一个作品）——混合在一起，使输出的图像看起来像内容图像， 但是用了风格参考图像的风格。

# 这是通过优化输出图像以匹配内容图像的内容统计数据和风格参考图像的风格统计数据来实现的。 这些统计数据可以使用卷积网络从图像中提取。

# 例如，我们选取这张小狗的照片和 Wassily Kandinsky 的作品 7：
# 瓦西里·康丁斯基（Василий Кандинский，格里历1866年12月16日－1944年12月13日），出生于俄罗斯的画家和美术理论家。与彼埃·蒙德里安和马列维奇一起，康定斯基认为是抽象艺术的先驱.

import tensorflow_hub as hub
import functools
import time
import PIL.Image
import numpy as np
import tensorflow as tf
import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


# 下载图像并选择风格图像和内容图像：
content_path = tf.keras.utils.get_file(
    'YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')

# https://commons.wikimedia.org/wiki/File:Vassily_Kandinsky,_1913_-_Composition_7.jpg
style_path = tf.keras.utils.get_file(
    'kandinsky5.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')


# 定义一个加载图像的函数，并将其最大尺寸限制为 512 像素。
def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

# 创建一个简单的函数来显示图像：


def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)


content_image = load_img(content_path)
style_image = load_img(style_path)

plt.subplot(1, 2, 1)
imshow(content_image, 'Content Image')

plt.subplot(1, 2, 2)
imshow(style_image, 'Style Image')

# 本教程演示了原始的风格迁移算法。其将图像内容优化为特定风格。在进入细节之前，让我们看一下 TensorFlow Hub 模块如何快速风格迁移：
hub_module = hub.load(
    'https://hub.tensorflow.google.cn/google/magenta/arbitrary-image-stylization-v1-256/1')
stylized_image = hub_module(tf.constant(
    content_image), tf.constant(style_image))[0]
tensor_to_image(stylized_image)

# 使用模型的中间层来获取图像的内容和风格表示。 从网络的输入层开始，前几个层的激励响应表示边缘和纹理等低级 feature (特征)。 随着层数加深，最后几层代表更高级的 feature (特征)——实体的部分，如轮子或眼睛。 在此教程中，我们使用的是 VGG19 网络结构，这是一个已经预训练好的图像分类网络。 这些中间层是从图像中定义内容和风格的表示所必需的。 对于一个输入图像，我们尝试匹配这些中间层的相应风格和内容目标的表示。
# 加载 VGG19 并在我们的图像上测试它以确保正常运行：
x = tf.keras.applications.vgg19.preprocess_input(content_image*255)
x = tf.image.resize(x, (224, 224))
vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
prediction_probabilities = vgg(x)
prediction_probabilities.shape
predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(
    prediction_probabilities.numpy())[0]
[(class_name, prob) for (number, class_name, prob) in predicted_top_5]
# 现在，加载没有分类部分的 VGG19 ，并列出各层的名称：
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

print()
for layer in vgg.layers:
    print(layer.name)
# 从网络中选择中间层的输出以表示图像的风格和内容：
# 内容层将提取出我们的 feature maps （特征图）
content_layers = ['block5_conv2']

# 我们感兴趣的风格层
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

# 使用tf.keras.applications中的网络可以让我们非常方便的利用Keras的功能接口提取中间层的值。

# 在使用功能接口定义模型时，我们需要指定输入和输出：

# model = Model(inputs, outputs)

# 以下函数构建了一个 VGG19 模型，该模型返回一个中间层输出的列表：


def vgg_layers(layer_names):
    # 加载我们的模型。 加载已经在 imagenet 数据上预训练的 VGG
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


# 然后建立模型
style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image*255)

# 查看每层输出的统计信息
for name, output in zip(style_layers, style_outputs):
    print(name)
    print("  shape: ", output.numpy().shape)
    print("  min: ", output.numpy().min())
    print("  max: ", output.numpy().max())
    print("  mean: ", output.numpy().mean())
    print()
# 图像的内容由中间 feature maps (特征图)的值表示。
# 事实证明，图像的风格可以通过不同 feature maps (特征图)上的平均值和相关性来描述。
# 通过在每个位置计算 feature (特征)向量的外积，并在所有位置对该外积进行平均,可以计算出包含此信息的 Gram 矩阵。
# 这可以使用tf.linalg.einsum函数来实现：


def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

# 构建一个返回风格和内容张量的模型。


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs*255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(
            inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (
            outputs[:self.num_style_layers], outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                        for style_output in style_outputs]

        content_dict = {content_name: value for content_name,
                        value in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value for style_name,
                    value in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}


# 在图像上调用此模型，可以返回 style_layers 的 gram 矩阵（风格）和 content_layers 的内容：
extractor = StyleContentModel(style_layers, content_layers)

results = extractor(tf.constant(content_image))

style_results = results['style']

print('Styles:')
for name, output in sorted(results['style'].items()):
    print("  ", name)
    print("    shape: ", output.numpy().shape)
    print("    min: ", output.numpy().min())
    print("    max: ", output.numpy().max())
    print("    mean: ", output.numpy().mean())
    print()

print("Contents:")
for name, output in sorted(results['content'].items()):
    print("  ", name)
    print("    shape: ", output.numpy().shape)
    print("    min: ", output.numpy().min())
    print("    max: ", output.numpy().max())
    print("    mean: ", output.numpy().mean())

# 梯度下降
# 使用此风格和内容提取器，我们现在可以实现风格传输算法。我们通过计算每个图像的输出和目标的均方误差来做到这一点，然后取这些损失值的加权和。

# 设置风格和内容的目标值：


style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']
# 定义一个 tf.Variable 来表示要优化的图像。
# 为了快速实现这一点，使用内容图像对其进行初始化（ tf.Variable 必须与内容图像的形状相同）


image = tf.Variable(content_image)
# 由于这是一个浮点图像，因此我们定义一个函数来保持像素值在 0 和 1 之间：


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
# 创建一个 optimizer 。 本教程推荐 LBFGS，但 Adam 也可以正常工作：


opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
# 为了优化它，我们使用两个损失的加权组合来获得总损失：


style_weight = 1e-2
content_weight = 1e4


def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                        for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2)
                            for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss
# 使用 tf.GradientTape 来更新图像。


@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))
# 现在，我们运行几个步来测试一下：


train_step(image)
train_step(image)
train_step(image)
tensor_to_image(image)


# 运行正常，我们来执行一个更长的优化：


start = time.time()

epochs = 10
steps_per_epoch = 100

step = 0
for n in range(epochs):
    for m in range(steps_per_epoch):
        step += 1
        train_step(image)
        print(".", end='')
    display.clear_output(wait=True)
    display.display(tensor_to_image(image))
    print("Train step: {}".format(step))

end = time.time()
print("Total time: {:.1f}".format(end-start))
