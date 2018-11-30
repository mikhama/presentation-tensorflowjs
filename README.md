# Presentation Draft

## Subject: "TensorFlow.js"

## Agenda

Introduction
1. History
    - What is TensorFlow?
        - History timeline (2011, 2015, 2017).
        - Languages support.
        - Platforms support.
    - And what about TensorFlow.js?
        - Client-Server Architecture.
        - 2017 Deeplearn.js.
        - Problem with speed. WebGL.
        - 2018 TensorFlow.js.
    - Who is using it?
2. Features
    - What are providing TensorFlow.js for Front-end Developers?
        - Working in browser.
        - Working at mobile devices.
        - Creating new models.
        - Running existing models.
        - Retrain existing models. 
3. Usage
    - But for what kind of applications I might use it?
        - Demos. 
        - Pacman. Training model.
        - Pacman. Playing!
        - Ideas about usage.
    - How I can getting started?
        - Import in a project.
    - TensorFlow.js runtime.
    - CoreAPI.
        - Tensors.
        - Variables.
        - Operations.
        - Memory management.
    - LayersAPI.
4. Hello TensorFlow.js
    - Creating from scratch and training simple model example.
        - Define the model.
        - Prepare model for training.
        - Give data to model.
        - Train model.
        - Use the model.
    - Using pre-trained model.
        - Using ready application.
        - Demonstrating the code.
Сonclusion

## Introduction

Hello everyone. My name is Dzmitry. And I'll tell you about TensorFlow.js.

Recently getting started with mashine learning was very expensive and time-consuming enjoyment. But today you can start with mashine learning very easily right in your browser, using JavaScript.

Before going to TensorFlow.js, I would like to start off with TensorFlow and its history.

## History

#### What is TensorFlow?

The official site of Tensor Flow https://www.tensorflow.org says: "This is an open source machine learning framework for everyone."

But earlier in 2011 it was developed at Google as their propriatory library for Mashine Learning applications at Google. It was called DistBelief.

And in November 2015 this library was open sourced under the Apache License.

In February 2017 Version 1.0 of TensorFlow is released.

It is a low-level C++ library with a lot of functionality for doing machine learning. In the world of data science Python is very popular, and it is primary language for TensorFlow. But TensorFlow works with many others programming languages, such as Java, C, Swift, Go, and of course JavaScript.

It can run on multiply CPUs and GPUs and is available on 64-bit Linux, macOS, Windows, and mobile computing platforms including Android and iOS.

#### Who is using it?

Tensor Flow has been using by many companies such as Google, Airbnb, Intel, AMD, UBER, and many others.

#### And what about TensorFlow.js?

In JavaScript Machine Learning was performed by using an API. An API was made using some framework, and the model was deployed at the server. The client sent a request using JavaScript to get results from the server.

In August 2017, a project called Deeplearn.js appeared, which aimed to enable Machine Learning in JavaScript, without the API hassle.

But there were questions about speed. It was very well known that JavaScript code could not run on GPU. To solve this problem, WebGL was introduced. This is a browser interface to OpenGL. WebGL enabled the execution of JavaScript code on GPU.

And in March 2018, the DeepLearn.js team got merged into the TensorFlow Team at Google and was renamed TensorFlow.js.

## Features

#### What are providing TensorFlow.js for Front-end Developers?

It entirety works in browser:
- It doesn't need drivers or installations.
- It is Highly interactive, because of JavaScript!

It runs at laptops and mobile devices that have sensors like the microphone, camera, accelerometer, etc.

With TesnsorFlow.js you can:
- Build and train models from scratch using the low-level JavaScript linear algebra library or the high-level layers API directly in the browser or under Node.js.
- Use TensorFlow.js model converters to run existing models (Keras) right in the browser or under Node.js.
- Retrain existing models using sensor data connected to the browser, or other client-side data.

### Usage

#### But for what kind of applications I might use it?

Official site of <a href="https://js.tensorflow.org/">TensorFlow.js</a> provides some demo applications available:

- EMOJI SCAVENGER HUNT is about finding objects in real world.
- WEBCAM CONTROLLER is about playing in Pacman.
- POSENET is about pose estimation.
- and others.

And I have made a gif-animation for demonstrate you example of amazing way playing in Pacman!

And now I show you how it easy to play in Packman with only head.

First of all I create some pre-trained examples for our model. I turn my head in up direction, then in left, right and down. And finally I press the button training. Our model is created!

You can see that pictogram with direction that I turn my head is highligted orange color when I turn my head.

And this is not a joke, I really don't use my hands!

I think this awesome possibility to use a controller that you want for different objectives.

I think you might to use TensorFlow.js for:
- Accessibility (For example, people with disabilities might to use head for switching pages).
- Games (Turn your imagination on...).
- Learning applications (Maybe for dancing learning application).
- and so on.

#### How I can getting started?

You can only add this line in your HTML-file:

`<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@0.12.0">`

Or you can install it as a Node.js package:

`npm install @tensorflow/tfjs`

And write it in your main js file:

`import * as tf from '@tensorflow/tfjs';`

#### TensorFlow.js runtime.

TensorFlow.js uses WebGL. And it provides two things:
- CoreAPI is a low level API for linear algebra and automatic differentation.
- LayerAPI is built over the CoreAPI, and makes our lives easier by increasing the level of abstraction.

TensorFlow supports models importing (TensorFlow and Keras).

#### CoreAPI

CoreAPI contains tools for creating and training models from scratch.

**Tensors**

So, what is a Tensor?

Tensor is a mathematical thing it's a structure that hold numbers in it.
I show you...

```
Scalar      1                // a scalar is a single number

Vector      [1 2]            // a vector is an array of numbers

Matrix      [1 2]            // a matrix is a two-dimensional array
            [3 4]

Tensor      [ [1 2] [3 4] ]  // a tensor is a n-dimensional array with n > 2
            [ [1 7] [5 4] ]
```

Tensors are the core datastructure of TensorFlow.js.
For creating a tensor you should use this method `tf.tensor(values, shape?, dtype?)`:

- **values** *(TypedArray|Array)* The values of the tensor. Can be nested array of numbers, or a flat array, or a TypedArray.
- **shape** *(number[])* The shape of the tensor. Optional. If not provided, it is inferred from values. Optional
- **dtype** *('float32'|'int32'|'bool'|'complex64')* The data type. Optional

**Variables**

Tensors are immutable data structures. This means their values can't be changed once they are set.

But if we need to change the data frequently then you should use `tf.variable()`:

```
const x = tf.variable(tf.tensor([1, 2, 3]));
x.assign(tf.tensor([4, 5, 6]));

x.print();
```

**Operations**

There are many operations (mathematic stuff) in TensorFlow.js you can find on the <a href="https://js.tensorflow.org/api/0.13.3/#Operations">documentaton page</a>. Operations always return new Tensors and newer modify input Tensors. But `tf.variable()` can be used in order to save memory.


`tf.add()` — Adds two tf.Tensors element-wise:

```
const a = tf.tensor1d([1, 2, 3, 4]);
const b = tf.tensor1d([10, 20, 30, 40]);

a.add(b).print();  // or tf.add(a, b)
// [11, 22, 33, 44]
```

`tf.matmul()` — Computes the dot product of two matrices, A * B. This operation is frequenlty used in Mashine Learning.

```
const a = tf.tensor2d([1, 2], [1, 2]);
const b = tf.tensor2d([1, 2, 3, 4], [2, 2]);

a.matMul(b).print();  // or tf.matMul(a, b)
```

**Memory Management**

Memory management is the key in the Machine Learning tasks, because they are generally computatuonally expensive.

TensorFlow.js provides two major ways to manage memory:

1. `tf.dispose()` - Disposes any `tf.Tensors` found within the provided object.
2. `tf.tidy()` - Using this method helps avoid memory leaks.

#### LayersAPI

Layers are the primary building block for constructing a Machine Learning model. Each layer will typically perform some computation to transform its input to its output. Under the hood, every layer uses the CoreAPI of TensorFlow.js.

### Hello TensorFlow.js

#### Creating from scratch and training simple model example.

It is an example from official site of <a href="https://js.tensorflow.org/">TensorFlow.js</a>.

```
<html>
  <head>
    <!-- Load TensorFlow.js -->
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@0.13.3/dist/tf.min.js"> </script>

    <!-- Place your code in the script tag below. You can also use an external .js file -->
    <script>
      // Notice there is no 'import' statement. 'tf' is available on the index-page
      // because of the script tag above.

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
        // Open the browser devtools to see the output
        model.predict(tf.tensor2d([5], [1, 1])).print();
      });
    </script>
  </head>

  <body>
  </body>
</html>
```

How it works?

For examle we have some graph. And we need to predict Y, when X = 5.

Steps:
1. Define a model.
2. Prepare the model for training: Specify the loss and the optimizer.
3. Generate some synthetic data for training.
4. Train the model using the data.
5. Use the model to do inference on a data point the model hasn't seen before.

With each new iteration a probability of prediction is increasing. Because model is training. `epochs` is a number of iterations.

```
epochs = 10 shows 7.3903484
epochs = 100 shows 8.130826
epochs = 1000 shows 8.9420271
```

It could have seemed very difficult because of some mathematics stuff.

#### Use pre-trained model.

I have created an application for <a href="https://mikhama.github.io/image-recognition/">image recognition</a> in order to show you that TensorFlow.js is not very hard to use on real life Web-Applications.

My application is very simple to use:
- Wait when model is loading.
- Push the button ang select an image.
- And see a result. The result is what kind of the object is on the image.
- You can see a probability, and five possible answers.

You can see the code on my <a href="https://github.com/mikhama/image-recognition">Github</a>.

I download ready for TensorFlow.js pre-trained MobileNet model.
You can download different models from official <a href="https://github.com/tensorflow/tfjs-models">Github</a> of TensorFlow.js. 

I have only 9 lines of code that are related with TensorFlow.js!

```
8   const model = await tf.loadModel('./model/model.json');
...

43  const offset = tf.scalar(127.5);
44  const tensor = tf.fromPixels(selectedImage)
45      .resizeNearestNeighbor([224, 224])
46      .toFloat()
47      .sub(offset)
48      .div(offset)
49      .expandDims();
50
51  const prediction = await model.predict(tensor).data();
```

First of all I load downloaded pre-trained model.

And then I do some magic with TensorFlow.js.

For good prediction recomended to do pre-processing of images. We need it, because images that are used for creating model and images that are used right now are different. They hue/saturation, color and brightness may be different. We have to normalize the image with use a different technics in order to make that particular real-word image into the same type of image what we have trained. That normalizing have been made from 45 to 49 lines.

### Conclusion

As a conclusion I would like to say that TensorFlow.js is very young, however it is robust and useful JavaScript-framework.
Thank you!
