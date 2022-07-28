#ifndef __NEURON__
#define __NEURON__

#include <vector>
#include <cmath>
#include <random>
#include "cpp-logger/logger.hpp"

class Neuron {
    /// Pointers to weights
    std::vector <float*> _input_ptrs;

    /// Pointers to input Neurons
    std::vector <Neuron*> _input_neuron_ptrs;

    /// Weights
    std::vector <float> _weights;

    /// The bias
    float _bias;

    /// The output
    float _output;

    /// Function pointer to activation function
    float (*_activation)(float sum);

    /// The simple activation function (Returns 0 or 1)
    static float activation_simple(float sum);

    /// The euler activation function (Returns from 0 to 1)
    static float activation_eul(float sum);

    /// Error
    float err;

public:

    /// Default constructor
    Neuron();

    /// Compy constructor
    Neuron(const Neuron &n);

    /// Getter for input (returns pointer NOT a copy of the vector)
    std::vector<float*>* get_input_ptrs();

    /// Getter for input neurons (returns pointer NOT a copy of the vector)
    std::vector<Neuron*>* get_input_neurons_ptrs();

    /// Getter for weights (returns pointer NOT a copy of the vector)
    std::vector<float>* get_weights_ptr();

    /// Setter for weights
    void set_weights(std::vector<float> &weights);
    void set_weights(int size);

    /// Getter for bias
    float get_bias();

    /// Setter for bias
    void set_bias(float value);

    /// Get for output value
    float get_output();

    /// Get output ptr
    float* get_output_ptr();

    /// Setter for input values
    void set_input_ptrs(std::vector<float*> &input);
    void set_input_ptrs(std::vector<float> &input);

    /// Get the activation index
    int get_activation_index();

    /// Set the activation function
    void set_activation_function(int index);

    /// Set input neurons
    void set_input_neurons_ptr(std::vector<Neuron> &input);     // With vector of neurons
    void set_input_neurons_ptr(std::vector<Neuron*> &input);    // With vector of pointers

    /// Randomize weights
    void randomize_weights();

    /// Getter for error
    float* get_error_ptr();

    /// Setter for error
    void set_error(float value);

    /// Evaluates the output of the neuron
    void evaluate();
};

#endif
