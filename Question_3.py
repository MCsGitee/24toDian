import numpy as np

def sdp_attention(Q, K, V):
    matmul_qk = np.matmul(Q, K.T) 
    d_k = Q.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(d_k)

    attention_weights = softmax(scaled_attention_logits)

    output = np.matmul(attention_weights, V)  
    return output, attention_weights

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def mh_attention(Q, K, V, num_heads):
    d_model = Q.shape[-1]
    assert d_model % num_heads == 0
    d_head = d_model // num_heads
  
    Q_split = np.reshape(Q, (Q.shape[0], Q.shape[1], num_heads, d_head))
    K_split = np.reshape(K, (K.shape[0], K.shape[1], num_heads, d_head))
    V_split = np.reshape(V, (V.shape[0], V.shape[1], num_heads, d_head))
    
    outputs = []
    for i in range(num_heads):
        Q_i = Q_split[:, :, i, :]  # Q_i (batch_size, seq_length, d_head)
        K_i = K_split[:, :, i, :]  # K_i (batch_size, seq_length, d_head)
        V_i = V_split[:, :, i, :]  # V_i (batch_size, seq_length, d_head)
        
        output, _ = sdp_attention(Q_i, K_i, V_i)
        outputs.append(output)
    
    # 拼接所有头的输出
    concat_output = np.concatenate(outputs, axis=-1)  # 连接各个头的输出
    return concat_output

# 测试随机数据
np.random.seed(0)
batch_size = 1
seq_length = 5
d_model = 8
num_heads = 2

Q = np.random.rand(batch_size, seq_length, d_model)
K = np.random.rand(batch_size, seq_length, d_model)
V = np.random.rand(batch_size, seq_length, d_model)

output = mh_attention(Q, K, V, num_heads)
print("output：\n", output)
