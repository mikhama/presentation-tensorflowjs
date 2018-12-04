# Presentation Draft

## Subject: "TensorFlow.js"

Link to task: https://github.com/rolling-scopes-school/tasks/blob/2018-Q3/tasks/presentation.md

## Agenda

- [Introduction](#introduction)
- [What is TensorFlow?](#what-is-tensorflow)
    - History timeline.
    - Languages support.
    - Platforms support.
- [And what about TensorFlow.js?](#and-what-about-tensorflowjs)
    - Client-Server Architecture.
    - August 2017 - Deeplearn.js - Problem with speed - WebGL.
    - March 2018 - TensorFlow.js.
- [Who is using it?](#who-is-using-it)
- [What is TensorFlow.js providing for Front-end Developers?](#what-is-tensorflowjs-providing-for-front-end-developers)
    - Working in a browser.
    - Working at mobile devices.
    - Creating new models.
    - Running existing models.
    - Retrain existing models. 
- [For what kind of applications I might use TensorFlow.js?](#for-what-kind-of-applications-i-might-use-tensorflowjs)
    - Demo applications. 
    - Pacman. Training model.
    - Pacman. Playing!
    - Ideas about usage.
- [How get started?](#how-get-started)
    - [Importing library in a project.](#how-get-started)
    - [TensorFlow.js runtime.](#tensorflowjs-runtime)
    - [Tensors.](#tensors)
    - [Important things.](#important-things)
- [Hello, TensorFlow.js!](#hello-tensorflowjs)
    - [Creating from scratch and training simple model.](#creating-from-scratch-and-training-simple-model)
    - [Using pre-trained model.](#use-pre-trained-model)
- [Сonclusion](#conclusion)

## Introduction

Hello everyone. My name is Dzmitry. I'll tell you about TensorFlow.js.

Recently getting started with machine learning was very expensive and time-consuming enjoyment, but today you can start with machine learning very easily right in your browser, using JavaScript.

Before going to TensorFlow.js, I would like to start off with TensorFlow.

## What is TensorFlow?

An <a href="https://www.tensorflow.org">official site</a> of Tensor Flow  says: "This is an open source machine learning framework for everyone."

Earlier in 2011, it was developed at Google as their proprietary library for Machine Learning applications. It was called DistBelief.

In November 2015 this library was open sourced under the Apache License.

In February 2017 version 1.0 of TensorFlow was released.

It is a low-level C++ library with a lot of functionality for doing Machine Learning.

In the world of data science, Python is very popular, and it is a primary language for TensorFlow. But TensorFlow works with many others programming languages, such as Java, C, Swift, Go, and of course JavaScript.

It can run on multiple CPUs and GPUs and is available on 64-bit Windows, macOS, Linux and mobile computing platforms including Android and iOS.

## And what about TensorFlow.js?

In JavaScript, Machine Learning was performed by using an API. A model was deployed at the server. A client sent a request on a server using JavaScript. Operations with the model were done on the server.
Then the server sent a response to the client.

In August 2017, a project called Deeplearn.js appeared, which aimed to enable Machine Learning in JavaScript, without the API hassle.

But there were questions about speed because JavaScript code couldn't run on GPU. To solve this problem, WebGL was introduced. This is a browser interface to OpenGL. WebGL enabled the execution of JavaScript code on GPU.

And in March 2018, the DeepLearn.js team got merged into the TensorFlow Team at Google and was renamed to TensorFlow.js.

## Who is using it?

Tensor Flow has been using by many companies such as Google, Airbnb, Intel and many others.

## What is TensorFlow.js providing for Front-end Developers?

It entirety works in a browser:
- It doesn't need drivers or installations.
- It is Highly interactive, because of JavaScript!

It runs at laptops and mobile devices that have sensors like the microphone, camera, accelerometer, etc.

With TesnsorFlow.js you can:
- Build and train models from scratch.
- Run existing models (TensorFlow or Keras) directly in the browser or under Node.js.
- Retrain existing models using sensor data connected to the browser, or other client-side data.

## For what kind of applications I might use TensorFlow.js?

Official site of <a href="https://js.tensorflow.org/">TensorFlow.js</a> provides some demo applications available:

- Emoji Scavenger Hunt is a game about finding objects in the real world.
- Webcam Controller is a game in Pacman.
- Teachable Machine lets you teach itself to recognize images and play songs.
- Performance RNN is a real-time piano performance by a neural network.
- Posenet is a human pose estimation in the browser.

And I have made a gif-animation for demonstrating you an example of the amazing way playing in Pacman!

And now I show you how it easy to play in Packman using only your head.

First of all, I create some pre-trained examples for our model. I turn my head in left direction, then in right, down and up. And finally, I press the button training. Our model is created!

You can see that pictogram with the direction that I turn my head is highlighted orange color when I turn my head.

This is not a joke, I really don't use my hands!

I think this awesome possibility to use a controller that you want for different objectives.

I think you might use TensorFlow.js for:
- Accessibility (For example, people who can't use hadns might use a head for switching pages).
- Games.
- Learning applications.
- and so on.

## How get started?

You can only add this line to your HTML-file:

`<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@0.12.0">`

Or you can install it as a Node.js package:

`npm install @tensorflow/tfjs`

And write it in your main js file:

`import * as tf from '@tensorflow/tfjs';`

### TensorFlow.js runtime

TensorFlow.js uses WebGL in a browser or computing unit (TPU, CPU, GPU) in Node.js. And it provides two things:
- CoreAPI is a low-level API for linear algebra and automatic differentiation. It contains tools for creating and training models from scratch.
- LayersAPI is the primary building block for constructing a Machine Learning model. Under the hood, every layer uses the CoreAPI of TensorFlow.js.

TensorFlow supports models importing (TensorFlow and Keras).

### Tensors

So, what is a Tensor?

```
Scalar      1                // a scalar is a single number

Vector      [1 2]            // a vector is an array of numbers

Matrix      [1 2]            // a matrix is a two-dimensional array
            [3 4]

Tensor      [ [1 2] [3 4] ]  // a tensor is a n-dimensional array with n > 2
            [ [1 7] [5 4] ]
```

Tensors are the core data structure of TensorFlow.js.
For creating a tensor you should use method `tf.tensor(values, shape?, dtype?)`:

### Important things

There are very important things that you are should know if you want to study machine learning with TensorFlow.js

**Variables.** Tensors are immutable data structures. But if you need to change it you should use method `tf.variable()`.

**Operations.** Operations always return new Tensors and newer modify input Tensors. TensorFLow.js has many operations such as addition, substraction, multipliying, dividing, finding dot product, etc.

**Memory Management.** Memory management is the key in the Machine Learning tasks because they are generally computationally expensive. TensorFlow.js provides two major ways to manage memory: `tf.dispose()` and `tf.tidy()`.

You can find more on the <a href="https://js.tensorflow.org/api/0.13.3/">documentation page</a>

## Hello, TensorFlow.js!

### Creating from scratch and training simple model

It is an example from the official site of <a href="https://js.tensorflow.org/">TensorFlow.js</a>.

For example, we have linear regression. And we need to predict "y" when "x" equals 5. 

We know that "y" will be 9 because we have some intellect.

Now let's teach the machine!

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
With each new iteration, a probability of prediction is increasing, because the model is training. `epochs` is a number of iterations.

You can see when "epochs" value is increasing, "y" is approximate to value that expected.

```
epochs = 10 shows 7.3903484
epochs = 100 shows 8.130826
epochs = 1000 shows 8.9420271
```

### Use pre-trained model.

I have created an application for <a href="https://mikhama.github.io/image-recognition/">image recognition</a> in order to show you that TensorFlow.js is not very hard to use on real life Web-Applications.

My application is very simple to use:
- Push the button and select an image.
- See a result. The result is what kind of the object is on the image.
- You can see a probability and five possible answers.

You can see the code on my <a href="https://github.com/mikhama/image-recognition">GitHub</a> page.

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

To create it, I have downloaded from the Internet ready for TensorFlow.js pre-trained MobileNet model.

I load this pre-trained model in constant. And then I take the image and make some pre-processing.
I need it because images that are used for creating model and images that are used right now are different. They hue/saturation, color and brightness may be different. I have to normalize the image by using a different technics.

## Conclusion

As a conclusion I would like to say that TensorFlow.js is very young, however, it is robust and useful JavaScript-framework.
Thank you!
