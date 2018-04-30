#include "nn.h"

//debug
int nb = 1;

std::vector<size_t> parseScheme(std::string scheme);
float getRand();

//Neuron
Neuron Neuron::initInputNeuron(float input) {
    this->output = input;
    this->connections = {};
    this->number = nb++;
    return *this;
}

Neuron Neuron::initNeuron(const std::vector<Neuron>& prev_layer_neurons) {
    this->number = nb++;
    this->output = 0.0f;
    for(size_t i = 0; i < prev_layer_neurons.size(); ++i){
        linkedNeuron ln;
        ln.neuron = prev_layer_neurons[i].getNum();
        ln.weight = getRand();
        this->connections.push_back(ln);
    }
    return *this;
}

Neuron Neuron::initBiasNeuron() {
    this->output = 1.0f;;
    this->number = nb++;
    this->connections = {};
    return *this;
}

float Neuron::calculate(const std::vector<Neuron>& prev_layer_neurons) {
    float result = 0.0f;
    for(size_t i = 0; i < this->connections.size(); ++i) {
        //std::cout << prev_layer_neurons[i].getOutput() << "*" << connections[i].weight << "+";
        result += prev_layer_neurons[i].getOutput() * connections[i].weight;
    }
    //std::cout << "=" << result << "\n";
    this->output = 1.0f / (1.0f + exp(-result));
    //std::cout << this->output << std::endl;
    return this->output;
}

//Layer
Layer Layer::initInputLayer(size_t layer_size, std::vector<float> inputs) {
    for(size_t i = 0; i < layer_size; ++i) {
        Neuron n;
        n = n.initInputNeuron(inputs[i]);
        this->neurons.push_back(n);
    }
    Neuron bias;
    bias = bias.initBiasNeuron();
    this->neurons.push_back(bias);
    return *this;
}

Layer Layer::initLayer(size_t layer_size, Layer& previous_layer) {
    for(size_t i = 0; i < layer_size; ++i) {
        Neuron n;
        const std::vector<Neuron>& vn = previous_layer.getNeurons();
        n = n.initNeuron(vn);
        this->neurons.push_back(n);
    }
    Neuron bias;
    bias.initBiasNeuron();
    this->neurons.push_back(bias);
    return *this;
}

const std::vector<Neuron>& Layer::getNeurons() const {
    return this->neurons;
}

std::vector<Neuron>& Layer::getNeurons() {
    return this->neurons;
}

float Layer::calcLayer(Layer &previous_layer) {
    float result = 0.0f;
    for(size_t i = 0; i < neurons.size() - 1; ++i) {
        result += neurons[i].calculate(previous_layer.getNeurons());
    }
    return result;
}

float Layer::calcError(std::vector<float> target) {
    float error = 0.0f;
    for(size_t i = 0; i < target.size(); ++i) {
        error += (pow((target[i] - neurons[i].getOutput()),2))/2;
    }
    return error;
}

//Network
Network Network::initNet(std::string scheme, std::vector<float> inputs) {
    std::vector<size_t> sch = parseScheme(scheme);
    Layer inp;
    inp.initInputLayer(sch[0], inputs);
    this->layers.push_back(inp);
    for(size_t i = 1; i < sch.size(); ++i) {
        Layer hid;
        hid.initLayer(sch[i], this->layers.back());
        this->layers.push_back(hid);
    }
    return *this;
}

std::vector<Neuron> Network::calcNet() {
    float result = 0.0f;
    for(size_t i = 1; i < this->layers.size(); ++i) {
        result = this->layers[i].calcLayer(this->layers[i-1]);
    }
    Layer last = this->layers.back();
    return last.getNeurons();
}

std::vector<size_t> parseScheme(std::string scheme) {
    std::vector<size_t> vscheme;
    for(size_t i = 0; i < scheme.length(); ++i) {
        if(isdigit(scheme[i])) {
            size_t t = scheme[i] - '0';
            vscheme.push_back(t);
        }
    }
    return vscheme;
}

void Network::print() { 
    for(auto a:this->layers) {
        std::cout << "\nLayer:";
        std::cout << "\n ls: " << a.getNeurons().size() << std::endl;
        std::vector<Neuron> h = a.getNeurons();
        for(auto b:h) {
            std::cout << "Neuron nr.: " << b.getNum() << " output: " << b.getOutput() << " connections: " << std::endl;
            for(auto t:b.getLN()) {
                std::cout << "connected to " << t.neuron << " and weight is: " << t.weight << std::endl;;
            }
            std::cout << std::endl;
        }
    }
}

float Network::calcError(std::vector<float> target) {
    float error = layers.back().calcError(target);
    return error;
}

void Network::printResults() {
    Layer last = layers.back();
    std::cout << "Results: ";
    for(std::vector<Neuron>::const_iterator sz = last.getNeurons().begin(); sz != last.getNeurons().end() - 1; ++sz) {
        std::cout << sz->getOutput() << " ";;
    }
    std::cout << std::endl;
}

void Network::setInputs(std::vector<float> inputs) {
    for(size_t i = 0; i < inputs.size(); ++i) {
        layers[0].getNeurons()[i].setOutput(inputs[i]);
    }
}


std::vector<float> Network::propogate(float target) {
    std::vector<float> deltas; 
    for(size_t i = 0; i < layers.back().getNeurons().back().getNum(); ++i) {
        deltas.push_back(0.0f);
    }
    std::vector<Neuron> vn = layers.back().getNeurons();
    for(std::vector<Neuron>::const_iterator it = vn.begin(); it != vn.end(); ++it) {
        float output = it->getOutput();
        float delta = output * (1 - output) * (output - target);
        deltas[it->getNum() - 1] = delta;
    }
    for(std::vector<Layer>::const_iterator it = layers.end() - 2; it != layers.begin(); --it) {
        std::vector<Neuron> lvn = it->getNeurons();
        for(std::vector<Neuron>::const_iterator nit = lvn.begin(); nit != lvn.end(); ++nit) {
            std::vector<Layer>::const_iterator prev_layer = it + 1;
            float output = nit->getOutput();
            float sumde = 0.0f;
            std::vector<Neuron> prev = prev_layer->getNeurons();
            for(std::vector<Neuron>::const_iterator lit = prev.begin(); lit != prev.end(); ++ lit) {
                std::vector<linkedNeuron> vlit = lit->getLN();
                for(std::vector<linkedNeuron>::const_iterator lnit = vlit.begin(); lnit != vlit.end(); ++lnit){
                    //std::cout << lnit->weight << " ";
                    //std::cout << deltas[lit->getNum() - 1] << std::endl;
                    if(lnit->neuron == nit->getNum()) {
                        sumde += lnit->weight * deltas[lit->getNum() - 1];
                        //std::cout << "\nSumde: " << sumde << std::endl;
                    }
                }
            }
            float delta = output * ( 1 - output) * sumde;
            deltas[nit->getNum() -1] = delta;
        }
    }
    /*
    for(auto i : deltas) {
        std::cout << i << ", ";
    }
    std::cout << std::endl;
    */
    return deltas;
}

void Network::updateWeights(const std::vector<float>& deltas, float lr) {
    for(std::vector<Layer>::iterator it = layers.begin() + 1; it != layers.end(); ++it) {
        std::vector<Neuron>& vn = it->getNeurons();
        for(std::vector<Neuron>::iterator nit = vn.begin(); nit != vn.end(); ++nit) {
            std::vector<linkedNeuron>& lnv = nit->getLN();
            for(std::vector<linkedNeuron>::iterator lit = lnv.begin(); lit != lnv.end(); ++lit) {
                std::vector<Neuron> prev = (it - 1)->getNeurons();
                float output = 0.0f;
                for(std::vector<Neuron>::iterator i = prev.begin(); i != prev.end(); ++i) {
                    if(lit->neuron == i->getNum()) {
                        output = i->getOutput();
                    }
                }
                lit->weight = lit->weight - lr * deltas[nit->getNum() - 1] * output;
            }
        }
    }
}

int main() {
    srand(time(NULL));
    float rlearn = 0.8f;
    float error = 0.0f;
    std::string scheme = "231";
    std::vector<std::vector<float>> inputs = {{ 0.0f, 0.0f }, { 1.0f, 0.0f }, { 0.0f, 1.0f }, { 1.0f, 1.0f }};
    std::vector<float> target = { 0.0f, 1.0f, 1.0f, 0.0f };
    Network network;
    network = network.initNet(scheme, inputs[0]);

    for(size_t i = 0; i < 18000; ++i) {
        for(size_t s = 0; s < inputs.size(); ++s) {
            network.setInputs(inputs[s]);
            std::vector<Neuron> results = network.calcNet();
            //error = network.calcError(target);
            std::vector<float> deltas = network.propogate(target[s]);
            network.updateWeights(deltas, rlearn);
           // std::cout << "Error: " << error << std::endl;
        }
    }
    network.print();
    network.printResults();

    /*
    //network.print();
    std::vector<Neuron> results = network.calcNet();
    //network.print();
    //network.printResults();
    float error = network.calcError(target);
    //std::cout << error << "<--" << std::endl;
    std::vector<float> deltas = network.propogate(error);
    network.updateWeights(deltas, rlearn);
    network.print();
    */
    return 0;
}

float getRand() {
    rand();
    rand();
    rand();
    return (rand()%100)/100.0;
}
