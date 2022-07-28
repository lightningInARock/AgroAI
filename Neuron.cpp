#include "Neuron.hpp"

#define e 2.7182818284

Neuron::Neuron() {
    _bias = 0;
    _activation = activation_simple;
}

Neuron::Neuron(const Neuron &n) {
    _input_ptrs = n._input_ptrs;
    _input_neuron_ptrs = n._input_neuron_ptrs;
    _weights = n._weights;
    _bias = n._bias;
    _output = n._output;
    _activation = n._activation;
}

float Neuron::activation_simple(float sum) {
    return sum > 0;
}

float Neuron::activation_eul(float sum) {
    return 1/(1 + pow(e, -sum));
}

std::vector<float*>* Neuron::get_input_ptrs() {
    return &_input_ptrs;
}

std::vector<Neuron*>* Neuron::get_input_neurons_ptrs() {
    return &_input_neuron_ptrs;
}

std::vector<float>* Neuron::get_weights_ptr() {
    return &_weights;
}

void Neuron::set_weights(std::vector<float> &weights) {
    _weights = weights;
}

void Neuron::set_weights(int size) {
    if(size <= 0) {
        Logger::error("Wrong weights size");
        exit(1);
    }

    _weights = std::vector<float> (size);
}

float Neuron::get_bias() {
    return _bias;
}

void Neuron::set_bias(float value) {
    _bias = value;
}

float Neuron::get_output() {
    return _output;
}

float* Neuron::get_output_ptr() {
    return &_output;
}

int Neuron::get_activation_index() {
    return (_activation == activation_eul);
}

void Neuron::set_activation_function(int index) {
    if(index) {
        _activation = activation_eul;
    } else {
        _activation = activation_simple;
    }
}

void Neuron::set_input_neurons_ptr(std::vector<Neuron> &input) {
    if(input.size() != _weights.size()) {
        Logger::error("Input size is not equal to weights size");
        exit(1);
    }

    _input_neuron_ptrs = std::vector<Neuron*> (input.size());
    _input_ptrs = std::vector<float*> (input.size());
    for(int i = 0; i < input.size(); ++i) {
        _input_neuron_ptrs[i] = &input[i];
        _input_ptrs[i] = input[i].get_output_ptr();
    }
}

void Neuron::set_input_neurons_ptr(std::vector<Neuron*> &input) {
    if(input.size() != _weights.size()) {
        Logger::error("Input size is not equal to weights size");
        exit(1);
    }

    _input_neuron_ptrs = input;
    _input_ptrs = std::vector<float*> (input.size());
    for(int i = 0; i < input.size(); ++i) {
        _input_ptrs[i] = input[i]->get_output_ptr();
    }
}

void Neuron::set_input_ptrs(std::vector<float*> &input) {
    if(input.size() != _weights.size()) {
        Logger::error("Input size is not equal to weights size");
        exit(1);
    }

    _input_ptrs = input;

}

void Neuron::set_input_ptrs(std::vector<float> &input) {
    if(input.size() != _weights.size()) {
        Logger::error("Input size is not equal to weights size");
        exit(1);
    }

    _input_ptrs = std::vector<float*> (input.size());
    for(int i = 0; i < input.size(); ++i) {
        _input_ptrs[i] = &input[i];
    }
}

void Neuron::randomize_weights() {
    for(int i = 0; i < _weights.size(); ++i) {
        _weights[i] = rand()/(float)RAND_MAX * 10 - 5; // to et value from -5 to 5
    }
}

float* Neuron::get_error_ptr() {
    return &err;
}

void Neuron::set_error(float value) {
    err = value;
}

void Neuron::evaluate() {
    float sum = _bias;
    for(int i = 0; i < _input_ptrs.size(); ++i) {
        sum += (*_input_ptrs[i])*_weights[i];
    }

    _output = _activation(sum);
}
