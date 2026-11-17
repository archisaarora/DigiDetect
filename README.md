# DigiDetect

### Learning how neural networks work by building one.

I wanted to understand what actually happens behind a neural network instead of just using an existing machine learning library and getting a prediction back.

So I built DigiDetect: a small handwritten digit recognition experiment that trains a neural network to recognize numbers from 0–9.

The idea was to build the model myself, experiment with its parameters, and get a better understanding of how training works from beginning to end.

## What I explored

- **Building a neural network from scratch:** Implemented a multilayer perceptron in Python (numpy for array maths only), including forward propagation, backpropagation, and gradient descent.
- **Understanding hidden layers:** Experimented with different hidden-layer sizes (32, 64, and 128 neurons) and explored how neurons combine input pixels into intermediate representations.
- **Training on MNIST:** Used the MNIST handwritten digit dataset, which contains 60,000 training images and 10,000 test images.
- **Evaluating performance:** Tracked training loss and test accuracy to see how the model changes as it learns.
- **Testing predictions:** Built a drawing interface where I can write my own digits and see what the model predicts.

## A little about how it works

Each image is represented as a 14 × 14 grid of pixel values, giving the network 196 inputs.

Those values pass through a hidden layer using ReLU activation. The output layer has 10 neurons, one for each possible digit, and uses softmax to produce prediction probabilities.

The model learns by comparing its predictions with the correct labels, calculating the error, and updating its weights and biases through backpropagation and mini-batch gradient descent.

I also added a separate section to the website explaining these concepts because I wanted to be able to follow what the code was actually doing, rather than treating the training process as a black box.

## Tech stack

- Python 3 + numpy
- HTML / CSS / a little JavaScript for the page
- MNIST dataset

No TensorFlow, PyTorch, or external machine learning framework is used for the neural network itself. Forward propagation, backpropagation and the weight updates are written by hand; numpy is only used for matrix arithmetic.

The original JavaScript version (the network running entirely in the browser) lives on the [`javascript`](../../tree/javascript) branch.

## Running it locally

Clone the repository:

```bash
git clone https://github.com/archisaarora/DigiDetect.git
cd DigiDetect
pip install numpy
```

Start it:

```bash
python -m digidetect
```

Your browser opens at `http://localhost:8000`. The Python server trains and runs the model; the page is the same interface as before.

The bundled sample dataset works offline. Loading the full MNIST dataset requires an internet connection the first time (it's cached in `~/.cache/digidetect` afterwards).

Prefer the terminal?

```bash
python -m digidetect train --hidden 128 --epochs 8 --full --save model.npz
```

## Running the tests

```bash
python -m unittest discover -s tests -v
```

The tests cover dataset decoding, prediction probabilities, gradient calculations (backprop vs. numerical gradients), loss reduction, model accuracy, the MNIST file parser, and the HTTP API the page uses.

## Project layout

| Path | What it is |
|---|---|
| `digidetect/nn.py` | the neural network: forward, backprop, momentum SGD |
| `digidetect/trainer.py` | batching, epochs, accuracy and loss history |
| `digidetect/data.py` | bundled sample + full MNIST download (pooled to 14 × 14) |
| `digidetect/server.py` | serves the site and exposes the model as a small JSON API |
| `index.html`, `css/`, `js/` | the website |

## Things I want to experiment with next

- How does increasing the number of hidden neurons affect accuracy?
- Does training on more data always improve the model?
- What happens when the model starts overfitting?
- How does a convolutional neural network compare to this basic fully connected network?
- Why does the model sometimes struggle with digits drawn by hand, even when it performs well on MNIST?

This project is an ongoing learning experiment. I built it to get more comfortable with the fundamentals of machine learning and to understand the math and implementation behind a model rather than relying entirely on prebuilt tools.