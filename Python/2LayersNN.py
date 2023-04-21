import math
import random


class NeuralNetwork:
    learning_rate = 0.5
    
    def __init__(self, num_inputs, num_hidden, num_outputs, Weights_IH=None, bias_IH=None,
                 Weights_HO=None, bias_HO=None):
        self.num_inputs = num_inputs
        
        self.hidden_layer = NeuronLayer(num_hidden, bias_IH)
        self.output_layer = NeuronLayer(num_outputs, bias_HO)
        
        self.init_Weights_IH(Weights_IH)
        self.init_Weights_HO(Weights_HO)
    
    def init_Weights_IH(self, Weights_IH):
        weight_num = 0
        for h in range(len(self.hidden_layer.neurons)):
            for i in range(self.num_inputs):
                if not Weights_IH:
                    self.hidden_layer.neurons[h].weights.append(random.random())
                else:
                    self.hidden_layer.neurons[h].weights.append(Weights_IH[weight_num])
                weight_num += 1
    
    def init_Weights_HO(self, Weights_HO):
        weight_num = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layer.neurons)):
                if not Weights_HO:
                    self.output_layer.neurons[o].weights.append(random.random())
                else:
                    self.output_layer.neurons[o].weights.append(Weights_HO[weight_num])
                weight_num += 1
    
    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')
    
    def feed_forward(self, inputs):
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        return self.output_layer.feed_forward(hidden_layer_outputs)
    
    def training(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)
        
        # 1. 输出神经元的值
        d_Loss_Y = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):
            # ∂E/∂zⱼ
            d_Loss_Y[o] = self.output_layer.neurons[
                o].d_Loss_H1(training_outputs[o])
        
        # 2. 隐含层神经元的值
        d_Loss_YH = [0] * len(self.hidden_layer.neurons)
        for h in range(len(self.hidden_layer.neurons)):
            
            # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
            d_Loss_YHO = 0
            for o in range(len(self.output_layer.neurons)):
                d_Loss_YHO += d_Loss_Y[o] * \
                              self.output_layer.neurons[o].weights[h]
            
            # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
            d_Loss_YH[h] = d_Loss_YHO * \
                           self.hidden_layer.neurons[
                               h].d_sigmoid()
        
        # 3. 更新输出层权重系数
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):
                # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
                d_Loss_Weight = d_Loss_Y[o] * self.output_layer.neurons[
                    o].d_H_W(w_ho)
                
                # Δw = α * ∂Eⱼ/∂wᵢ
                self.output_layer.neurons[o].weights[w_ho] -= self.learning_rate * d_Loss_Weight
        
        # 4. 更新隐含层的权重系数
        for h in range(len(self.hidden_layer.neurons)):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):
                # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                d_Loss_Weight = d_Loss_YH[h] * self.hidden_layer.neurons[
                    h].d_H_W(w_ih)
                
                # Δw = α * ∂Eⱼ/∂wᵢ
                self.hidden_layer.neurons[h].weights[w_ih] -= self.learning_rate * d_Loss_Weight
    
    def calculate_total_error(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])
        return total_error


class NeuronLayer:
    def __init__(self, num_neurons, bias):
        
        # 同一层的神经元共享一个偏置项b
        self.bias = bias if bias else random.random()
        
        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))
    
    def inspect(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])
            print('  Bias:', self.bias)
    
    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs
    
    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs


class Neuron:
    def __init__(self, bias):
        self.output = None
        self.inputs = None
        self.bias = bias
        self.weights = []
    
    def calculate_output(self, inputs):
        self.inputs = inputs
        self.output = self.sigmoid(self.calculate_sum())
        return self.output
    
    def calculate_sum(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias
    
    # 激活函数sigmoid
    def sigmoid(self, total_net_input):
        return 1 / (1 + math.exp(-total_net_input))
    
    def d_Loss_H1(self, target_output):
        return self.d_Loss_Output(target_output) * self.d_sigmoid();
    
    # 每一个神经元的误差是由平方差公式计算
    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2
    
    def d_Loss_Output(self, target_output):
        return -(target_output - self.output)
    
    def d_sigmoid(self):
        return self.output * (1 - self.output)
    
    def d_H_W(self, index):
        return self.inputs[index]


nn = NeuralNetwork(2, 2, 2, Weights_IH = [0.15, 0.2, 0.25, 0.3], bias_IH = 0.35,
                   Weights_HO = [0.4, 0.45, 0.5, 0.55], bias_HO = 0.6)
for i in range(10000):
    nn.training([0.05, 0.1], [0.01, 0.09])
    print(i, round(nn.calculate_total_error([[[0.05, 0.1], [0.01, 0.09]]]), 9))
