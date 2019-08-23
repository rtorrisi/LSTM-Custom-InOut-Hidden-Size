Since *PyTorch* ```torch.nn.LSTM``` and ```torch.nn.LSTMCell``` implementations only permit to specify a unique **hidden_size** for both input and output hidden state, I realized a custom LSTM model which *allow to give as model input an hidden_state which dimension can differ from the hidden_state that the model produce in output*.

This implementation could be usefull for specific case such as in **[Matching Networks for One Shot Learning](https://arxiv.org/pdf/1606.04080.pdf)**, where the concatenation of LSTM hidden output and another vector is required as input of the next LSTM step.

The only limitation of this implementation is that, if input and output hidden state differ, the sample _x_ must have a sequence_len of 1. If sequence is needed, it have to be managed externally. Alternatively, the user have to specify the way the hidden output of the first sequence element will be readjusted in order to process the next sequence element within the for-loop of the CustomLSTM class.

Here is an example of our CustomLSTM using a different hidden_size for input and output.

```python
in_batch_size = 5
in_sequence_len = 1 # read the description above
in_feature_size = 2

in_hidden_size = 6
out_hidden_size = 3

custom_lstm = CustomLSTM(input_size=in_feature_size, input_hidden_size=in_hidden_size, output_hidden_size=out_hidden_size)
```


```python
# generate a random input of shape [batch, sequence, feature]
random_input = torch.FloatTensor(in_batch_size, in_sequence_len, in_feature_size).normal_()

custom_output, (h_out, c_out) = custom_lstm(random_input, h_in=torch.zeros(in_hidden_size), c_in=torch.zeros(out_hidden_size))
custom_output, (h_out, c_out) = custom_lstm(random_input, h_in=torch.cat((h_out, h_out), 1), c_in=c_out)
```


```python
print(custom_output)
```

    tensor([[[ 0.1137, -0.0591, -0.0782]],
    
            [[ 0.3027, -0.0427, -0.0110]],
    
            [[ 0.1662, -0.1025, -0.0611]],
    
            [[ 0.2475, -0.0816, -0.0311]],
    
            [[ 0.1992, -0.0799, -0.0468]]], grad_fn=<TransposeBackward0>)


We can also verify that original LSTM and our custom LSTM produce the same result when hidden_state has the same input and output size 

```python
in_batch_size = 2
in_sequence_len = 3
in_feature_size = 2
in_hidden_size = 4
out_hidden_size = 4


# input weights and hidden weights
w_ih = torch.Tensor(out_hidden_size*4,in_feature_size); w_ih.data.normal_(0,0.1)
w_hh = torch.Tensor(out_hidden_size*4,in_hidden_size); w_hh.data.normal_(0,0.1)

# input bias and hidden bias (bias could be only once)
b_ih = torch.Tensor(out_hidden_size*4); b_ih.data.normal_(0,0.1)
b_hh = torch.Tensor(out_hidden_size*4); b_hh.data.normal_(0,0.1)

print(w_ih.shape, w_hh.shape, end="\n")
print(b_ih.shape, b_ih.shape, end="\n")
```

    torch.Size([16, 2]) torch.Size([16, 4])
    torch.Size([16]) torch.Size([16])
    


```python
original_lstm = torch.nn.LSTM(input_size=in_feature_size, hidden_size=out_hidden_size, num_layers=1, batch_first=True)
custom_lstm = CustomLSTM(input_size=in_feature_size, input_hidden_size=in_hidden_size, output_hidden_size=out_hidden_size)
```


```python
# we have to initialize original and custom LSTM weights and biases with the same values
original_lstm.weight_ih_l0 = torch.nn.Parameter(w_ih)
original_lstm.weight_hh_l0 = torch.nn.Parameter(w_hh)
original_lstm.bias_ih_l0 = torch.nn.Parameter(b_ih)
original_lstm.bias_hh_l0 = torch.nn.Parameter(b_hh)

custom_lstm.input_linear.weight = torch.nn.Parameter(w_ih)
custom_lstm.hidden_linear.weight = torch.nn.Parameter(w_hh)
custom_lstm.input_linear.bias = torch.nn.Parameter(b_ih)
custom_lstm.hidden_linear.bias = torch.nn.Parameter(b_hh)
```


```python
# generate a random input of shape [batch, sequence, feature]
random_input = torch.FloatTensor(in_batch_size, in_sequence_len, in_feature_size).normal_()

original_output, (h_out, c_out) = original_lstm(random_input)
custom_output, (h_out, c_out) = custom_lstm(random_input)
```


```python
print(original_output, custom_output, sep="\n\n-------------\n\n")
```

    tensor([[[-0.0527, -0.0105, -0.0484,  0.0221],
             [-0.0858, -0.0058, -0.0083,  0.0094],
             [-0.1109, -0.0073, -0.0423,  0.0239]],
    
            [[-0.0480, -0.0100, -0.0224,  0.0130],
             [-0.0601, -0.0248, -0.1676,  0.0772],
             [-0.0429, -0.0316, -0.2057,  0.0988]]], grad_fn=<TransposeBackward0>)
    
    -------------
    
    tensor([[[-0.0527, -0.0105, -0.0484,  0.0221],
             [-0.0858, -0.0058, -0.0083,  0.0094],
             [-0.1109, -0.0073, -0.0423,  0.0239]],
    
            [[-0.0480, -0.0100, -0.0224,  0.0130],
             [-0.0601, -0.0248, -0.1676,  0.0772],
             [-0.0429, -0.0316, -0.2057,  0.0988]]], grad_fn=<CopyBackwards>)    
