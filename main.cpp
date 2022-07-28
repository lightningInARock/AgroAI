// #include "Neuron.hpp"
#include "NeuralNetwork.hpp"

int main() {

    NeuralNetwork AgroAI;

    AgroAI.set_input_layer(16384);
    Logger::info("Set the input layer with size: 16384");

    AgroAI.add_hidden_layer(1000);
    Logger::info("Added first hidden layer with size: 1000");

    AgroAI.add_hidden_layer(1000);
    Logger::info("Added second hidden layer with size: 1000");

    AgroAI.set_output_layer(10, 1);
    Logger::info("Set output layer with size: 10");

//     AgroAI.load("AgroAI.aistate");
//     Logger::info("AI state loaded");


    std::fstream input_file;
    std::string filename = "../xaa";

     input_file.open(filename);
     std::string line;
     getline(input_file, line);
     float sup_val = line[line.size() - 1] - '0';
     line.pop_back();
     line.pop_back();
     line.pop_back();
     while(1) {
         AgroAI.learn(line, sup_val);
     }


    while(true) {
        input_file.open(filename);

        if(!input_file.is_open()) {
            Logger::error("File not found");
            break;
        }
        std::string line;
        while(getline(input_file, line)) {

            float sup_val = line[line.size() - 1] - '0';
            line.pop_back();
            line.pop_back();
            line.pop_back();
            AgroAI.learn(line, sup_val);
        }
        Logger::info("File completed");
        ++filename[4];
        input_file.close();
        AgroAI.save();
    }
    AgroAI.save();
    Logger::info("AI state saved");
    return 0;
}
