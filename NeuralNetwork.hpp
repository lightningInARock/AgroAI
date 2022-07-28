#ifndef __NEURAL_NETWORK__
#define __NEURAL_NETWORK__

#include <string>
#include <vector>
#include "Neuron.hpp"
#include "cpp-logger/logger.hpp"

class NeuralNetwork {
    /// The input layer
    std::vector <float> _input_layer;

    /// Hidden layers
    std::vector <std::vector<Neuron>> _hidden_layers;

    /// The output layer
    std::vector <Neuron> _output_layer;
public:

    /// Default constructor
    NeuralNetwork() {

    }

    /// Copy constructor
    NeuralNetwork(const NeuralNetwork &n) {
        this->_input_layer = n._input_layer;
        this->_hidden_layers = n._hidden_layers;
        this->_output_layer = n._output_layer;
    }


    /// Build input layer from size (empty by default)
    void set_input_layer(int size) {
        if(size <=0 ) {
            Logger::error("Wrong input layer size");
            exit(1);
        }

        _input_layer = std::vector<float> (size);
    }

    /// Setter for input layer
    void set_input_layer(std::vector<float> &input) {
        _input_layer = input;
    }

    /// Add hidden layer (define activation)
    void add_hidden_layer(int size, int activation_index=0) {
        if(size <= 0) {
            Logger::error("Invalid layer size");
            exit(1);
        }

        std::vector<Neuron> layer(size);
        for(int i = 0; i < size; ++i) {
            if(_hidden_layers.empty()) {
                if(_input_layer.empty()) {
                    Logger::error("Not pecified size of input layer");
                    exit(1);
                } else {
                    layer[i].set_weights(_input_layer.size());
                    layer[i].randomize_weights();
                    layer[i].set_input_ptrs(_input_layer);
                }
            } else {
                layer[i].set_weights(_hidden_layers[_hidden_layers.size() - 1].size());
                layer[i].randomize_weights();
                layer[i].set_input_neurons_ptr(_hidden_layers[_hidden_layers.size() - 1]);
            }
            layer[i].set_activation_function(activation_index);
            layer[i].randomize_weights();
        }

        _hidden_layers.push_back(layer);
    }

    /// Add hidden layer (argument is layer)
    void add_hidden_layer(std::vector <Neuron> &layer, int activation_index=0) {
        for(int i = 0; i < layer.size(); ++i) {
            if(_hidden_layers.empty()) {
                if(_input_layer.empty()) {
                    Logger::error("Not pecified size of input layer");
                    exit(1);
                } else {
                    layer[i].set_input_ptrs(_input_layer);
                }
            } else {
                layer[i].set_input_neurons_ptr(_hidden_layers[_hidden_layers.size() - 1]);
            }
            layer[i].set_activation_function(activation_index);
        }

        _hidden_layers.push_back(layer);
    }

    /// Setter for output layer (define activation)
    void set_output_layer(int size, int activation_index=0) {
        if(size <= 0) {
            Logger::error("Invalid layer size");
            exit(1);
        }

        _output_layer = std::vector<Neuron>(size);
        for(int i = 0; i < size; ++i) {
            if(_hidden_layers.size()) {
                _output_layer[i].set_weights(_hidden_layers[_hidden_layers.size() - 1].size());
                _output_layer[i].randomize_weights();
                _output_layer[i].set_input_neurons_ptr(_hidden_layers[_hidden_layers.size() - 1]);
            } else {
                if(_input_layer.empty()) {
                    Logger::warning("Not pecified size of input layer");
                } else {
                    _output_layer[i].set_weights(_input_layer.size());
                    _output_layer[i].randomize_weights();
                    _output_layer[i].set_input_ptrs(_input_layer);
                }
            }
            _output_layer[i].set_activation_function(activation_index);
        }
    }

    void set_output_layer(std::vector<Neuron> &layer) {
        _output_layer = layer;
        for(int i = 0; i < _output_layer.size(); ++i) {
            _output_layer[i].set_input_neurons_ptr(_hidden_layers[_hidden_layers.size() - 1]);
        }
    }

    /// Set the last hidden layer as output
    void set_last_hidden_layer_as_output() {
        _output_layer = _hidden_layers[_hidden_layers.size() - 1];
        _hidden_layers.pop_back();
    }

    /// Evaluate the output from given input
    std::vector<float> run(std::string input) {
        if(input.size() != _input_layer.size()) {
            Logger::error("Given input size is not equal to NeuralNetwork's input layer size");
            exit(1);
        }

        for(int i = 0; i < input.size(); ++i) {
            _input_layer[i] = input[i] - '0';
        }

        std::vector<float> res(_output_layer.size());
        for(int hid_idx = 0; hid_idx < _hidden_layers.size(); ++hid_idx) {
            for(int neur_idx = 0; neur_idx < _hidden_layers[hid_idx].size(); ++neur_idx) {
                _hidden_layers[hid_idx][neur_idx].evaluate();
            }
        }

        for(int i = 0; i < _output_layer.size(); ++i) {
            _output_layer[i].evaluate();
            res[i] = _output_layer[i].get_output();
        }

        return res;
    }


    /// Save the NeuralNetwork (aka save weights)
    void save() {
        std::fstream file;
        file.open("AgroAI.aistate", std::ios::out);
        if(!file.is_open()) {
            Logger::error("Couldn't open file to save");
            exit(1);
        }

        file << _input_layer.size() << std::endl;

        if(_output_layer.empty()) {
            Logger::error("Output layer is not set");
            exit(1);
        }

        for(int hid_idx = 0; hid_idx < _hidden_layers.size(); ++hid_idx) {
            file << _hidden_layers[hid_idx][0].get_activation_index() << "*\n";
            for(int neur_idx = 0; neur_idx < _hidden_layers[hid_idx].size(); ++neur_idx) {
                file << _hidden_layers[hid_idx][neur_idx].get_bias() << '|';
                std::vector <float>* weights = _hidden_layers[hid_idx][neur_idx].get_weights_ptr();
                Logger::warning(std::to_string(weights->size()));
                for(int i = 0; i < weights->size(); ++i) {
                    file << (*weights)[i] << ' ';
                }
                file << '\n';
            }
            file << '_' << std::endl;
            Logger::info("Saved layer with input size: " + std::to_string(_hidden_layers[hid_idx][0].get_input_ptrs()->size()));
        }

        file << _output_layer[0].get_activation_index() << "*\n";
        for(int i = 0; i < _output_layer.size(); ++i) {
            file << _output_layer[i].get_bias() << '|';
            std::vector <float>* weights = _output_layer[i].get_weights_ptr();
            for(int i = 0; i < weights->size(); ++i) {
                file << (*weights)[i] << ' ';
            }
            file << std::endl;
        }
        file << "=" << std::endl;
        Logger::info("Saved output layer with size: " + std::to_string(_output_layer[0].get_input_neurons_ptrs()->size()));
    }

    /// Load the NeuralNetwork (aka et weights and layers)
    void load(std::string filename) {
        std::fstream file;
        file.open(filename, std::ios::in);
        if(!file.is_open()) {
            Logger::error("Couldn't open file to save");
            exit(1);
        }

        _hidden_layers.clear();
        _output_layer.clear();

        std::vector<Neuron> layer;
        std::string line;

        getline(file, line);

        set_input_layer(std::stoi(line));
        Logger::info("Input layer size loaded: " + line);

        while(getline(file, line)) {
            Neuron n;
            std::vector <float> weights;
            int activation_index = 0;
            if(line == "_") {
                Logger::info("Weights count = " + std::to_string(layer[0].get_weights_ptr()->size()));
                add_hidden_layer(layer, activation_index);
                Logger::info("Loaded layer with size: " + std::to_string(layer.size()));
                layer.clear();
            } else if(line == "=") {
                set_output_layer(layer);
            } else if(line[line.size() - 1] == '*') {
                activation_index = std::stoi(line.substr(0, line.size() - 1));
            } else {
                std::string num;
                for(int i = 0; i < line.size(); ++i) {
                    if(line[i] == ' ') {
                        if(!num.empty()) {
                            weights.push_back(std::stof(num));
                        } else {
                            Logger::warning("Empty value in file");
                        }
                        num.clear();
                    } else if(line[i] == '|') {
                        n.set_bias(std::stof(num));
                        num.clear();
                    } else {
                        num.push_back(line[i]);
                    }
                }
                n.set_weights(weights);
                layer.push_back(n);
            }
        }
        set_last_hidden_layer_as_output();
    }

    /// Learn from example
    void learn(std::string input, float sup_value) {                            // *** NOT WORKING, NEEDS TO BE FIXED ***
        std::vector<float> res = run(input);
        std::string print_value;
        for(int i = 0; i < res.size(); ++i) {
            print_value += std::to_string(res[i]*100).substr(0, 6) + "% ";
        }
        Logger::info("NeuralNetwork output: " + print_value);

        std::vector<float> sup (res.size());
        ++sup[sup_value];

        /// Error back propogation
        float learn_coef = 1.0f;



        float network_error = 0;
        for(int i = 0; i < sup.size(); ++i) {
            network_error += (res[i] - sup[i])*(res[i] - sup[i]);
        }


        // Output layer
        for(int i = 0; i < sup.size(); ++i) {
            float diff = res[i] - sup[i];    // C = (res - sup)
            float err = 2 * diff;
            err *= (_output_layer[i].get_activation_index())? res[i] * (1 - res[i]) : 0.1;
            _output_layer[i].set_error(err);
            std::vector<float>* weights = _output_layer[i].get_weights_ptr();
            std::vector<float*>* inputs = _output_layer[i].get_input_ptrs();
            for(int j = 0; j < weights->size(); ++j) {
                float gradient = *(*inputs)[i] * err;
                (*weights)[i] -= learn_coef*gradient;
            }
        }

        // 2(err) * res(1-res)

        // Hidden layers
        for(int l = _hidden_layers.size() - 1; l > 0; --l) {
            for(int i = 0; i < _hidden_layers[l].size(); ++i) {
                // Evaluate error
                float diff = 0;
                if(l == _hidden_layers.size() - 1) {
                    for(int o = 0; o < _output_layer.size(); ++o) {
                        float tmp = 2 * (res[o] - sup[o]);
                        tmp *= (_output_layer[o].get_activation_index())? res[i] * (1 - res[i]) : 0.1;
                        tmp *= (*_output_layer[o].get_weights_ptr())[i];
                        diff += tmp;
                    }
                } else {
                    float tmp = 0;
                    for(int o = 0; o < sup.size(); ++o) {
                        tmp = 2 * (res[o] - sup[o]);
                        tmp *= (_output_layer[o].get_activation_index())? res[o] * (1 - res[o]) : 0.1;
                    }
                    for(int f = 0; f < _hidden_layers[l + 1].size(); ++f) {
                        tmp *= (*_hidden_layers[l + 1][f].get_weights_ptr())[i];
                        diff += tmp;
                    }
                }
                _hidden_layers[l][i].set_error(diff);

                std::vector<float>* weights = _hidden_layers[l][i].get_weights_ptr();
                std::vector<float*>* inputs = _hidden_layers[l][i].get_input_ptrs();
                for(int j = 0; j < weights->size(); ++j) {
                    // float gradient = *(*inputs)[i] * 2 * diff;
                    (*weights)[i] -= learn_coef * diff; //gradient
                }
            }
        }
    }

};

#endif
