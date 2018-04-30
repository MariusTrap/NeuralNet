#ifndef NN_H
#define NN_H

#include<iostream>
#include<string>
#include<vector>
#include<cstdlib>
#include<ctime>
#include<cmath>

struct linkedNeuron {
    int neuron;
    float weight;
};

class Neuron {
    public:
    Neuron() { this->number = 0; this->output = 0.0f; this->connections = {}; }
    Neuron initInputNeuron(float input);
    Neuron initNeuron(const std::vector<Neuron>& prev_layer_neurons);
    Neuron initBiasNeuron();
    float calculate(const std::vector<Neuron>& prev_layer_neurons);

    float getOutput() const { return this->output; }
    void setOutput(float output) { this->output = output; }
    const std::vector<linkedNeuron>& getLN() const { return this->connections; }
    std::vector<linkedNeuron>& getLN() { return this->connections; }

    //void setNum(int num) { this->number = num; }
    int getNum() const { return this->number; }

    private:
    int number;
    float output;
    std::vector<linkedNeuron> connections;
};

class Layer {
    public:
    Layer() { this->neurons = {}; }
    Layer initInputLayer(size_t layer_size, std::vector<float> inputs);
    Layer initLayer(size_t layer_size, Layer &previous_layer);
    const std::vector<Neuron>& getNeurons() const;
    std::vector<Neuron>& getNeurons();
    float calcLayer(Layer& previous_layer);

    float calcError(std::vector<float> target);

    private:
    std::vector<Neuron> neurons;
};

class Network {
    public:
    Network() { this->layers = {}; }
    Network initNet(std::string scheme, std::vector<float> inputs);
    std::vector<Neuron> calcNet();
    float calcError(std::vector<float> target);
    void printResults();
    void setInputs(std::vector<float> inputs);

    std::vector<float> propogate(float target);

    void updateWeights(const std::vector<float>& deltas, float lr);

    void print();

    private:
    std::vector<Layer> layers;
};

#endif
