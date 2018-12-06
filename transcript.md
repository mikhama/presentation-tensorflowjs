# TensorFlow.js

## Introduction
 
Hello everyone. My name is Dzmitry, and I'll tell you about TensorFlow.js.

Recently getting started with Machine Learning was very expensive and time-consuming enjoyment, but today you can start with Machine Learning very easily right in your browser, using JavaScript.

But before going to TensorFlow.js, I would like to start off with TensorFlow.

## What is TensorFlow?

An official site says: "This is an open source machine learning framework for everyone."

But in 2011, it was developed at Google as their proprietary library for Machine Learning applications. It was called DistBelief.

In 2015 it was open sourced under the Apache License.

In 2017 version 1.0 of TensorFlow was released.

It is a low-level C++ library with a lot of functionality for doing Machine Learning.

In the world of data science, Python is very popular language, and it is also a primary language for TensorFlow. But TensorFlow works with many others languages, such as Java, C, Go, Swift, and of course JavaScript.

It can be run on multiple CPUs and GPUs and is available on 64-bit desktop platforms such as Windows, macOS, Linux and mobile devices including Android and iOS.

## And what about TensorFlow.js?

In JavaScript, Machine Learning was performed by using an API. The model was deployed at the server. The client sent a request to the server. Server made some operations with model, and sent a response back to the client.

In August 2017, a project called Deeplearn.js appeared, which aimed to enable Machine Learning in JavaScript, without the API hassle.

But there were questions about speed because JavaScript code couldn't run on GPU. To solve this problem, WebGL was introduced. WebGL is a browser interface to OpenGL. It executes JavaScript code on GPU.

And in March 2018, the DeepLearn.js team got merged into the TensorFlow Team at Google and it was renamed to TensorFlow.js.

## Who is using it?

Tensor Flow is using by many companies such as Google, Intel, Airbnb and many others.

## What is TensorFlow.js providing for Front-end Developers?

It entirety works in a browser:
- It doesn't need any drivers or installations.
- It is Highly interactive, because of JavaScript!

It can be run on laptops and mobile devices that have sensors like the camera, accelerometer, or microphone and so on.

With TesnsorFlow.js you can build and train models from scratch. You can also run existing models, such as TensorFlow models or Keras models under Node.js or right in your browser. You can retrain existing models using sensor data connected to the browser, or other client-side data.

## For what kind of applications I might use TensorFlow.js?

Official site provides some demo applications:

- Emoji Scavenger Hunt is a game about finding objects in the real world.
- Webcam Controller is a game in Pacman.
- Teachable Machine lets you teach itself to recognize images and play songs.
- Performance RNN is a real-time piano performance by a neural network.
- Posenet is a human pose estimation in a browser.

And I have made a gif-animation in order to show you how it easy to play in Pacman using only your head.

And we need to setup a controller.

First of all, I create some examples for our model. I turn my head in left direction, then in right, down and up. And after that, I press the button "Train model", and our model is created!

When our controller is ready, we are ready to play!

And you can see: when I turn my head in some direction the pictogram with that direction highlighted orange color. And this is not a joke, I really don't use my hands!

And I think you might use TensorFlow.js for:
- Accessibility (For example, people who can't use hands might use a head in order to switching pages in your Web-application).
- Or you can use it for games or learning applications.
- and so on.

## How get started?

You can only add one line to your HTML-file.

Or you can install it as a Node.js package, and import it in your main js file.

TensorFlow.js uses WebGL in a browser or processing unit (TPU, CPU, GPU) in Node.js. And it provides two things:
- CoreAPI is a low-level API for linear algebra and automatic differentiation. It contains tools for creating and training models from scratch.
- LayersAPI is the primary building block for constructing a Machine Learning model. Under the hood, every layer uses the CoreAPI.

TensorFlow.js supports models importing. You can import models such as Keras models or TensorFlow models into your project.

So, what is a Tensor?

A scalar is a single number, a vector is an array of numbers, a matrix is a two-dimensional array, and a tensor is a n-dimensional array with n > 2.

Tensor is a primary data structure in TensorFLow.

There are very important things that you are should know if you would like to start studying machine learning with TensorFlow.js:

Tensors are immutable data structures. But you can use method `variable` in order to change it.

Operations never modify input Tensors and always return new Tensors. TensorFLow.js provides many operations such as addition, substraction, multipliying, dividing, finding dot product, and so on.

Memory management is the key in the Machine Learning tasks because they are generally computationally expensive. For memory management you can use methods: `dispose` and `tidy`.

You can see more on the documentation page.

## Hello, TensorFlow.js!

### Creating from scratch and training simple model

For example, we have linear regression.

And we need to predict "y" when "x" equals 5. We know that "y" will be 9 because we have some intellect. Now let's teach the machine!

1. Define a model for linear regression.
2. Prepare the model for training.
3. Generate some synthetic data for training.
4. Train the model using the data.
5. Use the model.

Our model is training with each iteration. `epochs` is a number of these iterations. 
 You can see when "epochs" value is increasing, our "y" value is approximate to value that expected.

### Use pre-trained model.

I have created a simple application in order to show you that using TensoFlow for real-life applications is not very hard to use.

My application is very simple to use: you can select an image, and you can see a result. The result is what kind of the object is on the image. And you also can see probablity and five answers.

There is my code example.

First of all a downloaded from the Internet MobileNet model, and I loaded it into my script.

And then need some pre-processing of the image. I need pre-processing, because images that was using for creating the model and images that I use right now are different. They color, brightness hue/saturation maybe different, and I use different technics in order to normalize that.

You can see the whole on my GitHub page.

## Conclusion

And I think that TensorFlow.js is very young, however, it is very robust and useful framework.

Thank you!
