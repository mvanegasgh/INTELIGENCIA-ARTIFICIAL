import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimensión de cada cabeza

        # Proyecciones lineales W_Q, W_K, W_V y proyección final W_O (sec. 3.2 y 3.3)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, seq_len, d_model = x.size()

        # 1. Proyectar y dividir en cabezas:
        # (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        Q = self.W_q(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 2. Atención escalada: softmax(Q K^T / sqrt(d_k)) V
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        pesos = F.softmax(scores, dim=-1)
        atencion = pesos @ V

        # 3. Concatenar las cabezas: de vuelta a (batch, seq_len, d_model)
        atencion = atencion.transpose(1, 2).contiguous().view(batch, seq_len, d_model)

        # 4. Proyección final W_O
        return self.W_o(atencion)


if __name__ == "__main__":
    d_model, num_heads = 64, 8
    batch, seq_len = 4, 10

    mha = MultiHeadAttention(d_model, num_heads)
    X = torch.randn(batch, seq_len, d_model)  # tensor sintético
    salida = mha(X)

    print("Entrada:", list(X.shape))
    print("Salida: ", list(salida.shape))
    print("Dimensiones conservadas:", X.shape == salida.shape)