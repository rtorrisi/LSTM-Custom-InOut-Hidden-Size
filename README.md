```python
import torch
```


```python
class CustomLSTM(torch.nn.Module):
    def __init__(self, input_size, input_hidden_size, output_hidden_size):
        super(CustomLSTM, self).__init__()
        self.input_size = input_size
        self.input_hidden_size = input_hidden_size
        self.output_hidden_size = output_hidden_size
        
        self.input_linear = torch.nn.Linear(input_size, output_hidden_size*4, bias=True)
        self.hidden_linear = torch.nn.Linear(input_hidden_size, output_hidden_size*4, bias=True)
        
    def forward(self, x, h_in=None, c_in=None):
        '''
        x of shape [batch, sequence, feature]
        h_in of shape [input_hidden_size]
        c_in shape [output_hidden_size]
        '''
        batch_size, seq_size, feature_size = x.size()
        hidden_seq = []
        
        if h_in is None:
            h_in = torch.zeros(self.input_hidden_size)
        if c_in is None:
            c_in = torch.zeros(self.output_hidden_size)
        
        o_h_s = self.output_hidden_size
        
        for seq in range(seq_size):
            seq_x = x[:, seq, :]
            
            input_linear_out = self.input_linear(seq_x)
            hidden_linear_out = self.hidden_linear(h_in)
            
            gates = input_linear_out + hidden_linear_out
            
            input_gate = torch.sigmoid(gates[:, :o_h_s])
            forget_gate = torch.sigmoid(gates[:, o_h_s:o_h_s*2]) 
            candidate_gate = torch.tanh(gates[:, o_h_s*2:o_h_s*3])
            output_gate = torch.sigmoid(gates[:, o_h_s*3:])

            c_in = c_out = (forget_gate * c_in) + (input_gate * candidate_gate)
            h_in = h_out = (output_gate * torch.tanh(c_out))
            hidden_seq.append(h_out.unsqueeze(0))
           
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from [sequence, batch, feature] to [batch, sequence, feature]
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        
        return hidden_seq, (h_out, c_out)
```

Firstly, we have to verify that original LSTM and our custom LSTM produce the same result when hidden_state has the same input and output size 


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
    

Now try our custom LSTM using a a different size for in_hidden_size and out_hidden_size. Sequence length have to be of size 1


```python
in_batch_size = 5
in_sequence_len = 1
in_feature_size = 2

in_hidden_size = 6
out_hidden_size = 3

custom_lstm = CustomLSTM(input_size=in_feature_size, input_hidden_size=in_hidden_size, output_hidden_size=out_hidden_size)
```


```python
# generate a random input of shape [batch, sequence, feature]
rin = torch.FloatTensor(in_batch_size, in_sequence_len, in_feature_size).normal_()

custom_output, (h_out, c_out) = custom_lstm(rin, h_in=torch.zeros(in_hidden_size), c_in=torch.zeros(out_hidden_size))
custom_output, (h_out, c_out) = custom_lstm(rin, h_in=torch.cat((h_out, h_out), 1), c_in=c_out)
```


```python
print(custom_output)
```

    tensor([[[ 0.1137, -0.0591, -0.0782]],
    
            [[ 0.3027, -0.0427, -0.0110]],
    
            [[ 0.1662, -0.1025, -0.0611]],
    
            [[ 0.2475, -0.0816, -0.0311]],
    
            [[ 0.1992, -0.0799, -0.0468]]], grad_fn=<TransposeBackward0>)
    
