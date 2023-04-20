#include<bits/stdc++.h>
using namespace std;

class NeuralNetwork {
private:
	int epochs;
	double learning_rate;
	double w1, w2, w3, w4, w5, w6;
	double b1, b2, b3;
public:
	NeuralNetwork(int es, double lr) : epochs(es), learning_rate(lr) {
		w1 = w2 = w3 = w4 = w5 = w6 = 0;
		b1 = b2 = b3 = 0;
	}
	double forward(vector<double> data);
	void training(vector<vector<double>> data, vector<double>label);
	void predict(vector<vector<double>>test_data, vector <double>test_label);
	double sigmoid(double x);
	double deriv_sigmoid(double x);
	double getMSEcost(double pred, double label);
};
double NeuralNetwork::forward(vector<double> data) {
	double sum_h1 = w1 * data[0] + w2 * data[1] + b1;
	double sum_h2 = w3 * data[0] + w4 * data[1] + b2;
	double sum_y1 = w5 * sigmoid(sum_h1) + w6 * sigmoid(sum_h2) + b3;
	double y1 = sigmoid(sum_y1);
	return y1;
}

double NeuralNetwork::deriv_sigmoid(double x) {
	double y = sigmoid(x);
	return y * (1 - y);
}
double NeuralNetwork::sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

void NeuralNetwork::predict(vector<vector<double>>data, vector<double>label) {
	double cnt = 0;
	for (int i = 0; i < data.size(); i++) {
		double pred = forward(data[i]);
		pred = pred > 0.5 ? 1 : 0;
		if (label[i] == pred) {
			cnt++;
		}
	}
	cout << "correct rate:" << cnt / data.size() << endl;
}
double NeuralNetwork::getMSEcost(double pred, double label) {
	return (pred - label) * (pred - label);
}
void NeuralNetwork::training(vector<vector<double>>data, vector<double>label) {
	for (int epoch = 0; epoch < epochs; ++epoch) {
		for (int i = 0; i < data.size(); ++i) {
			vector<double> x = data[i];
			double sum_h1 = x[0] * w1 + x[1] * w2 + b1;
			double sum_h2 = x[0] * w3 + x[1] * w4 + b2;
			double h1 = sigmoid(sum_h1);
			double h2 = sigmoid(sum_h2);
			double sum_y1 = h1 * w5 + h2 * w6 + b3;
			double y1 = sigmoid(sum_y1);
			double pred = y1;
			double d_loss_pred = -2 * (label[i] - pred);

			double d_pred_h1 = w5 * deriv_sigmoid(sum_y1);
			double d_pred_h2 = w6 * deriv_sigmoid(sum_y1);

			double d_pred_w5 = h1 * deriv_sigmoid(sum_y1);
			double d_pred_w6 = h2 * deriv_sigmoid(sum_y1);
			double d_pred_b3 = deriv_sigmoid(sum_y1);

			double d_h1_w1 = x[0] * deriv_sigmoid(sum_h1);
			double d_h1_w2 = x[1] * deriv_sigmoid(sum_h1);
			double d_h1_b1 = deriv_sigmoid(sum_h1);

			double d_h2_w3 = x[0] * deriv_sigmoid(sum_h2);
			double d_h2_w4 = x[1] * deriv_sigmoid(sum_h2);
			double d_h2_b2 = deriv_sigmoid(sum_h2);
			w1 -= learning_rate * d_loss_pred * d_pred_h1 * d_h1_w1;
			w2 -= learning_rate * d_loss_pred * d_pred_h1 * d_h1_w2;
			w3 -= learning_rate * d_loss_pred * d_pred_h2 * d_h2_w3;
			w4 -= learning_rate * d_loss_pred * d_pred_h2 * d_h2_w4;
			w5 -= learning_rate * d_loss_pred * d_pred_w5;
			w6 -= learning_rate * d_loss_pred * d_pred_w6;
			b1 -= learning_rate * d_loss_pred * d_pred_h1 * d_h1_b1;
			b2 -= learning_rate * d_loss_pred * d_pred_h2 * d_h2_b2;
			b3 -= learning_rate * d_loss_pred * d_pred_b3;
		}
		if (epoch % 10 == 0) {
			double loss = 0;
			for (int i = 0; i < data.size(); i++) {
				loss += getMSEcost(forward(data[i]), label[i]);
			}
			cout << "epoch:" << epoch << " loss:" << loss << endl;
		}
	}
	cout << w1 << " " << w2 << " " << w3 << " " << w4 << " " << w5 << " " << w6 << " " << endl;
}

int main() {
	NeuralNetwork NN = NeuralNetwork(1000, 0.1);
	vector<vector<double>>training_data = { {-2,-1},{25,6},{17,4},{-15,-6} };
	vector<double>training_label = { 1,0,0,1 };
	vector<vector<double>>test_data = { {-3,-4},{-5,-4},{12,3},{-13,-4},{9,12} };
	vector<double>test_label = { 1,1,0,1,0 };
	NN.training(training_data, training_label);
	NN.predict(test_data, test_label);
	return 0;
}

