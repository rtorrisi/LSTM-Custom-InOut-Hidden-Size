import torch

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
