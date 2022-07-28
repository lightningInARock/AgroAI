# AgroAI
Simple Neural Network

This is a simple NeuralNetwork creating library

Just create an object of a class NeuralNetwork
```C++
  NeuralNetwork AgroAI;
```

You can use any name. In this example I used AgroAI

Start creating layers with special methoods
```C++
  AgroAI.set_input_layer(16384);
  Logger::info("Set the input layer with size: 16384");

  AgroAI.add_hidden_layer(1000);
  Logger::info("Added first hidden layer with size: 1000");

  AgroAI.add_hidden_layer(1000);
  Logger::info("Added second hidden layer with size: 1000");

  AgroAI.set_output_layer(10, 1);
  Logger::info("Set output layer with size: 10");
```

If NO additional argument given, it sets the activation function to -> (x > 0)? 1 : 0;
If you give additional argument, it sets the activation function to 1/(1 + e^(-x))

Use "learn" method to learn from example

Use save methood to save Neural Network weights

Use load methood to laod Neural Network weights (if you have already created one)
F. E.
```C++
  NeuralNetwork AgroAI;
  AgroAI.load("weights");
```
It will automaticly create all the layers
