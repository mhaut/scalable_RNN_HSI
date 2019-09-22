# Scalable Recurrent Neural Network for Hyperspectral Image Classification
The Code for "Scalable Recurrent Neural Network for Hyperspectral Image Classification". []
```
M. E. Paoletti, J. M. Haut, J. Plaza and A. Plaza.
Scalable Recurrent Neural Network for Hyperspectral Image Classification.
Journal of Supercomputing.
```

![ROhsi](https://github.com/mhaut/scalable_RNN_HSI/blob/master/images/RNN_for_HSI.png)



## Example of use
### Download datasets

```
sh retrieveData.sh
```

### Run code

```
Indian Pines
python -u rnn.py --dataset IP # Proposed
python -u rnn.py --dataset IP --vanillarnn # Vanilla RNN CUDA
python -u rnn.py --dataset IP --vanillarnn --cudnn # Vanilla RNN CuDNN
python -u rnn.py --dataset IP --lstm # LSTM CUDA
python -u rnn.py --dataset IP --lstm --cudnn # LSTM CuDNN
python -u rnn.py --dataset IP --gru # GRU CUDA
python -u rnn.py --dataset IP --gru --cudnn # GRU CuDNN
University of Pavia
python -u rnn.py --dataset PU # Proposed
python -u rnn.py --dataset PU --vanillarnn # Vanilla RNN CUDA
python -u rnn.py --dataset PU --vanillarnn --cudnn # Vanilla RNN CuDNN
python -u rnn.py --dataset PU --lstm # LSTM CUDA
python -u rnn.py --dataset PU --lstm --cudnn # LSTM CuDNN
python -u rnn.py --dataset PU --gru # GRU CUDA
python -u rnn.py --dataset PU --gru --cudnn # GRU CuDNN
Salinas Valley
python -u rnn.py --dataset SV # Proposed
python -u rnn.py --dataset SV --vanillarnn # Vanilla RNN CUDA
python -u rnn.py --dataset SV --vanillarnn --cudnn # Vanilla RNN CuDNN
python -u rnn.py --dataset SV --lstm # LSTM CUDA
python -u rnn.py --dataset SV --lstm --cudnn # LSTM CuDNN
python -u rnn.py --dataset SV --gru # GRU CUDA
python -u rnn.py --dataset SV --gru --cudnn # GRU CuDNN
```

