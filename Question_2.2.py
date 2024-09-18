import numpy as np

def RoPE(input_tensor, seq_length, dim):
    
    assert dim % 2 == 0, 

    m = dim // 2
    theta = 1.0 / (10000 ** (np.arange(0, m) / m))
    position = np.arange(seq_length)
    angle = np.einsum('i,j->ij', position, theta)

    sin_dim_pos = np.sin(angle)
    cos_dim_pos = np.cos(angle)

    even_input = input_tensor[:, :, 0::2]
    odd_input = input_tensor[:, :, 1::2]

    even_output = even_input * cos_dim_pos - odd_input * sin_dim_pos
    odd_output = even_input * sin_dim_pos + odd_input * cos_dim_pos

    output = np.zeros_like(input_tensor)
    output[:, :, 0::2] = even_output
    output[:, :, 1::2] = odd_output

    return output
