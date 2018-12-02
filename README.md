# Presentation Draft

## Subject: "TensorFlow.js"

## Agenda

- [Introduction](#intro)
- [What is TensorFlow?](#link1)
    - History timeline.
    - Languages support.
    - Platforms support.
- [And what about TensorFlow.js?](#link2)
    - Client-Server Architecture.
    - August 2017 - Deeplearn.js - Problem with speed - WebGL.
    - March 2018 - TensorFlow.js.
- [Who is using it?](#link3)
- [What is TensorFlow.js providing for Front-end Developers?](#link4)
    - Working in a browser.
    - Working at mobile devices.
    - Creating new models.
    - Running existing models.
    - Retrain existing models. 
- [For what kind of applications I might use TensorFlow.js?](#link5)
    - Demo applications. 
    - Pacman. Training model.
    - Pacman. Playing!
    - Ideas about usage.
- [How get started?](#link6)
    - [Importing library in a project.](#link6)
    - [TensorFlow.js runtime.](#link7)
    - [Tensors.](#link8)
    - [Variables.](#link9)
    - [Operations.](#link10)
    - [Memory management.](#link11)
- [Hello, TensorFlow.js!](#link12)
    - [Creating from scratch and training simple model.](#link13)
    - [Using pre-trained model.](#link14)
- [Сonclusion](#link15)

## <a name="#intro"></a> Introduction

Hello everyone. My name is Dzmitry. I'll tell you about TensorFlow.js.

Recently getting started with machine learning was very expensive and time-consuming enjoyment, but today you can start with machine learning very easily right in your browser, using JavaScript.

Before going to TensorFlow.js, I would like to start off with TensorFlow.

## <a name="#link1"></a> What is TensorFlow?

An <a href="https://www.tensorflow.org">official site</a> of Tensor Flow  says: "This is an open source machine learning framework for everyone."

Earlier in 2011, it was developed at Google as their proprietary library for Machine Learning applications at Google. It was called DistBelief.

In November 2015 this library was open sourced under the Apache License.

In February 2017 version 1.0 of TensorFlow is released.

It is a low-level C++ library with a lot of functionality for doing Machine Learning. In the world of data science, Python is very popular, and it is a primary language for TensorFlow. But TensorFlow works with many others programming languages, such as Java, C, Swift, Go, and of course JavaScript.

It can run on multiple CPUs and GPUs and is available on 64-bit Linux, macOS, Windows, and mobile computing platforms including Android and iOS.

## <a name="#link2"></a> And what about TensorFlow.js?

In JavaScript, Machine Learning was performed by using an API. An API was made using some framework, and the model was deployed at the server. The client sent a request using JavaScript to get results from the server.

In August 2017, a project called Deeplearn.js appeared, which aimed to enable Machine Learning in JavaScript, without the API hassle.

But there were questions about speed. It was very well known that JavaScript code could not run on GPU. To solve this problem, WebGL was introduced. This is a browser interface to OpenGL. WebGL enabled the execution of JavaScript code on GPU.

And in March 2018, the DeepLearn.js team got merged into the TensorFlow Team at Google and was renamed to TensorFlow.js.

## <a name="#link3"></a> Who is using it?

Tensor Flow has been using by many companies such as Google, Airbnb, Intel, AMD, UBER, and many others.

## <a name="#link4"></a> What is TensorFlow.js providing for Front-end Developers?

It entirety works in a browser:
- It doesn't need drivers or installations.
- It is Highly interactive, because of JavaScript!

It runs at laptops and mobile devices that have sensors like the microphone, camera, accelerometer, etc.

With TesnsorFlow.js you can:
- Build and train models from scratch using the low-level JavaScript linear algebra library or the high-level layers API directly in the browser or under Node.js.
- Use TensorFlow.js model converters to run existing models (Keras) right in the browser or under Node.js.
- Retrain existing models using sensor data connected to the browser, or other client-side data.

## <a name="#link5"></a> For what kind of applications I might use TensorFlow.js?

Official site of <a href="https://js.tensorflow.org/">TensorFlow.js</a> provides some demo applications available:

- Emoji Scavenger Hunt is a game about finding objects in the real world.
- Webcam Controller is a game about playing in Pacman.
- Teachable Machine lets you teach itself to recognize images and play songs.
- Performance RNN is a real-time piano performance by a neural network.
- Posenet is a human pose estimation in the browser.

And I have made a gif-animation for demonstrating you an example of the amazing way playing in Pacman!

And now I show you how it easy to play in Packman using only your head.

First of all, I create some pre-trained examples for our model. I turn my head in up direction, then in left, right and down. And finally, I press the button training. Our model is created!

You can see that pictogram with the direction that I turn my head is highlighted orange color when I turn my head.

This is not a joke, I really don't use my hands!

I think this awesome possibility to use a controller that you want for different objectives.

I think you might use TensorFlow.js for:
- Accessibility (For example, people with disabilities might use a head for switching pages).
- Games (Turn your imagination on...).
- Learning applications (Maybe for dancing learning application).
- and so on.

## <a name="#link6"></a> How get started?

You can only add this line to your HTML-file:

`<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@0.12.0">`

Or you can install it as a Node.js package:

`npm install @tensorflow/tfjs`

And write it in your main js file:

`import * as tf from '@tensorflow/tfjs';`

### <a name="#link7"></a> TensorFlow.js runtime

TensorFlow.js uses WebGL. And it provides two things:
- CoreAPI is a low-level API for linear algebra and automatic differentiation. It contains tools for creating and training models from scratch.
- LayersAPI is the primary building block for constructing a Machine Learning model. Under the hood, every layer uses the CoreAPI of TensorFlow.js.

TensorFlow supports models importing (TensorFlow and Keras).

### <a name="#link8"></a> Tensors

So, what is a Tensor?

Tensor is a mathematical thing it's a structure that holds numbers in it.

```
Scalar      1                // a scalar is a single number

Vector      [1 2]            // a vector is an array of numbers

Matrix      [1 2]            // a matrix is a two-dimensional array
            [3 4]

Tensor      [ [1 2] [3 4] ]  // a tensor is a n-dimensional array with n > 2
            [ [1 7] [5 4] ]
```

Tensors are the core data structure of TensorFlow.js.
For creating a tensor you should use this method `tf.tensor(values, shape?, dtype?)`:

- **values** *(TypedArray|Array)* The values of the tensor. Can be nested array of numbers, or a flat array, or a TypedArray.
- **shape** *(number[])* The shape of the tensor. Optional. If not provided, it is inferred from values. Optional.
- **dtype** *('float32'|'int32'|'bool'|'complex64')* The data type. Optional.

### <a name="#link9"></a> Variables

Tensors are immutable data structures. This means their values can't be changed once they are set.

But if we need to change the data frequently then you should use `tf.variable()`:

```
const x = tf.variable(tf.tensor([1, 2, 3]));
x.assign(tf.tensor([4, 5, 6]));

x.print();
```

### <a name="#link10"></a> Operations

There are many operations in TensorFlow.js you can find on the <a href="https://js.tensorflow.org/api/0.13.3/#Operations">documentation page</a>. Operations always return new Tensors and newer modify input Tensors. But `tf.variable()` can be used in order to save memory.

`tf.add()` — Adds two `tf.Tensors` element-wise:

```
const a = tf.tensor1d([1, 2, 3, 4]);
const b = tf.tensor1d([10, 20, 30, 40]);

a.add(b).print();  // or tf.add(a, b)
// [11, 22, 33, 44]
```

`tf.matmul()` — Computes the dot product of two matrices, A * B. This operation is frequently used in Machine Learning.

```
const a = tf.tensor2d([1, 2], [1, 2]);
const b = tf.tensor2d([1, 2, 3, 4], [2, 2]);

a.matMul(b).print();  // or tf.matMul(a, b)
```

### <a name="#link11"></a> Memory Management

Memory management is the key in the Machine Learning tasks because they are generally computationally expensive.

TensorFlow.js provides two major ways to manage memory:

1. `tf.dispose()` - Disposes any `tf.Tensors` found within the provided object.
2. `tf.tidy()` - Using this method helps avoid memory leaks.

## <a name="#link12"></a> Hello, TensorFlow.js!

### <a name="#link13"></a> Creating from scratch and training simple model

It is an example from the official site of <a href="https://js.tensorflow.org/">TensorFlow.js</a>.

```
// Define a model for linear regression.
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

// Prepare the model for training: Specify the loss and the optimizer.
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// Generate some synthetic data for training.
const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

// Train the model using the data.
model.fit(xs, ys, {epochs: 1000}).then(() => {
// Use the model to do inference on a data point the model hasn't seen before:
model.predict(tf.tensor2d([5], [1, 1])).print();
});
```

For example, we have linear regression. And we need to predict Y when X = 5.

Steps:
1. Define a model.
2. Prepare the model for training: Specify the loss and the optimizer.
3. Generate some synthetic data for training.
4. Train the model using the data.
5. Use the model to do inference on a data point the model hasn't seen before.

With each new iteration, a probability of prediction is increasing, because the model is training. `epochs` is a number of iterations.

```
epochs = 10 shows 7.3903484
epochs = 100 shows 8.130826
epochs = 1000 shows 8.9420271
```

### <a name="#link14"></a> Use pre-trained model.

I have created an application for <a href="https://mikhama.github.io/image-recognition/">image recognition</a> in order to show you that TensorFlow.js is not very hard to use on real life Web-Applications.

My application is very simple to use:
- Wait when a model is loading.
- Push the button and select an image.
- And see a result. The result is what kind of the object is on the image.
- You can see a probability and five possible answers.

You can see the code on my <a href="https://github.com/mikhama/image-recognition">Github</a>.

```
8   const model = await tf.loadModel('./model/model.json');
...
43  const offset = tf.scalar(127.5);
44  const tensor = tf.fromPixels(selectedImage)
45    .resizeNearestNeighbor([224, 224])
46    .toFloat()
47    .sub(offset)
48    .div(offset)
49    .expandDims();
50
51  const prediction = await model.predict(tensor).data();
```

I have only 10 lines of code that are related to TensorFlow.js!

I download from the Internet ready for TensorFlow.js pre-trained MobileNet model.

First of all, I load this pre-trained model.

And then I do some magic with TensorFlow.js.

For good prediction recommended doing pre-processing of images. We need it because images that are used for creating model and images that are used right now are different. They hue/saturation, color and brightness may be different. We have to normalize the image by using a different technics in order to make that particular real-world image into the same type of image what we have trained. That normalizing have been made from 45 to 49 lines.

## <a name="#link15"></a> Conclusion

As a conclusion I would like to say that TensorFlow.js is very young, however, it is robust and useful JavaScript-framework.
Thank you!
