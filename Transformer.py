import torch
import torch.nn as nn
from typing import Optional

class Transformer(nn.Module):
    def __init__(
                    self,
                    vocab_size:int,
                    d_model:int,
                    enc_context_size:int,
                    dec_context_size:int,
                    pos_enc_dropout: float,
                    num_encoder_blocks: int,
                    num_decoder_blocks: int,
                    num_heads: int,
                    drop_prob: float,
                    d_feedfwd: int,
                    pre_norm: bool
                ):
        super(Transformer, self).__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pre_norm = pre_norm
        self.num_encoder_blocks = num_encoder_blocks
        self.num_decoder_blocks = num_decoder_blocks

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        torch.nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

        self.enc_pos_encoding = nn.Embedding(enc_context_size, d_model)
        self.dec_pos_encoding = nn.Embedding(dec_context_size, d_model)

        torch.nn.init.normal_(self.enc_pos_encoding.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.dec_pos_encoding.weight, mean=0.0, std=0.01)

        self.encoder = nn.Sequential(*[EncoderBlock (
                                                    num_heads, 
                                                    enc_context_size, 
                                                    d_model, 
                                                    drop_prob, 
                                                    d_ff = d_feedfwd, 
                                                    ) for _ in range(num_encoder_blocks)])

        self.decoder = nn.Sequential(*[DecoderBlock(
                                                        num_heads,
                                                        dec_context_size,
                                                        d_model,
                                                        drop_prob,
                                                        d_ff = d_feedfwd
                                                    ) for _ in range(num_decoder_blocks)])
        
        if pre_norm:
            self.enc_final_ln = nn.LayerNorm(d_model)
            self.dec_final_ln = nn.LayerNorm(d_model)
        
        self.apply(self.init_linear_weights)

        self.head = nn.Linear(d_model, vocab_size)

        self.token_embedding.weight = self.head.weight
        
        self.head.bias.data.zero_()

    def init_linear_weights(self, module):
        if isinstance(module, ResidualScaledLinear):
            if module.is_encoder:
                torch.nn.init.normal_(module.weight, mean=0.0, std = 0.02 * ((2 * self.num_encoder_blocks) ** -0.5))
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std = 0.02 * ((3 * self.num_decoder_blocks) ** -0.5))

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, X, Y, context = None):

        enc_token_emb = self.token_embedding(context)
        enc_pos_enc = self.enc_pos_encoding(context)
        context = enc_token_emb + enc_pos_enc

        context = self.encoder(context, self.pre_norm)
        
        if self.pre_norm:
            context = self.enc_final_ln(context)
        
        dec_token_emb = self.token_embedding(X)
        dec_pos_enc = self.dec_pos_encoding(X)
        X = dec_token_emb + dec_pos_enc

        X = self.decoder(X, context, self.pre_norm)
        
        if self.pre_norm:
            X = self.dec_final_ln(X)
        
        X = self.head(X)

        return X

class Decoder(nn.Module):
    def __init__(
                    self, 
                    vocab_size:int,
                    d_model:int,
                    context_size:int,
                    pos_enc_dropout: float,
                    num_decoder_blocks: int,
                    num_heads: int,
                    drop_prob: float,
                    d_feedfwd: int,
                    pre_norm: bool,
                    mask_attention: bool
                ):
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pre_norm = pre_norm
        self.num_decoder_blocks = num_decoder_blocks

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        torch.nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

        self.pos_encoding = nn.Embedding(context_size, d_model)
        torch.nn.init.normal_(self.pos_encoding.weight, mean=0.0, std=0.01)

        self.decoder = nn.ModuleList([OnlyDecoderBlock(
                                                        num_heads,
                                                        context_size,
                                                        d_model,
                                                        drop_prob,
                                                        d_ff = d_feedfwd,
                                                        mask_attention = mask_attention
                                                    ) for _ in range(num_decoder_blocks)])

        if pre_norm:
            self.dec_final_ln = nn.LayerNorm(d_model)
        
        self.apply(self.init_linear_weights)
            
        self.head = nn.Linear(d_model, vocab_size)

        self.token_embedding.weight = self.head.weight
        
        self.head.bias.data.zero_()

    def init_linear_weights(self, module):
        if isinstance(module, ResidualScaledLinear):
            torch.nn.init.normal_(module.weight, mean=0.0, std = 0.02 * ((2 * self.num_decoder_blocks) ** -0.5))

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, X, attention_mask: Optional[torch.FloatTensor] = None):

        pos = torch.arange(0, X.size(1), dtype=torch.long, device=X.device)
        dec_pos_enc = self.pos_encoding(pos)
        dec_token_emb = self.token_embedding(X)
        X = dec_token_emb + dec_pos_enc
        
        for db in self.decoder:
            X = db(X, self.pre_norm, attention_mask)
        
        if self.pre_norm:
            X = self.dec_final_ln(X)
        
        X = self.head(X)

        return X

class PositionalEncoding(nn.Module):
    def __init__(
                    self, 
                    context_size: int, 
                    d_model :int,
                    drop_prob :float = 0.1
                ):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(drop_prob)

        pos_enc = torch.zeros(context_size, d_model)

        position = torch.arange(0, context_size, dtype=torch.float)[:, None]
        
        denominator = torch.pow(torch.tensor(10000), torch.arange(0, d_model, 2, dtype=torch.float) / d_model)

        pos_enc[:, 0::2] = torch.sin(position / denominator)
        pos_enc[:, 1::2] = torch.cos(position / denominator)

        self.register_buffer('pos_enc', pos_enc)

    def forward(self, X):
        X = self.pos_enc[:X.size(1)]
        return X

class ResidualScaledLinear(nn.Linear):
    def __init__(self, in_features, out_features, is_encoder, bias=True):
        super(ResidualScaledLinear, self).__init__(in_features, out_features, bias)
        self.is_encoder = is_encoder

class ScaledDotProductAttention(nn.Module):
    def __init__(
                    self,
                    context_size: int,
                    d_model: int,
                    dim_keys: int,
                    dim_values: int,
                    drop_prob: float,
                    mask_attention: bool
                ):
        super(ScaledDotProductAttention, self).__init__()
        
        self.dim_keys = dim_keys
        self.mask_attention = mask_attention

        self.queries = nn.Linear(d_model, dim_keys)
        self.keys = nn.Linear(d_model, dim_keys)
        self.values = nn.Linear(d_model, dim_values)

        self.softmax = nn.Softmax(-1)
        self.dropout = nn.Dropout(drop_prob)

        if mask_attention:
            self.register_buffer('mask', torch.tril(torch.ones(context_size, context_size)))

    def forward(self, X_Q, X_KV, attention_mask: Optional[torch.FloatTensor] = None):

        Q = self.queries(X_Q)
        K = self.keys(X_KV)

        out = (Q @ K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.dim_keys))
        
        if self.mask_attention and attention_mask == None:
            out = out.masked_fill(self.mask[:X_Q.size(1), :X_Q.size(1)] == 0, float('-inf'))
        elif self.mask_attention and attention_mask != None:
            out = out.masked_fill((attention_mask.unsqueeze(-2) * self.mask[:X_Q.size(1), :X_Q.size(1)]) == 0, float('-inf'))
        
        out = self.softmax(out)
        out = self.dropout(out)
        V = self.values(X_KV)
        out = out @ V

        return out

class MultiHeadAttention(nn.Module):
    def __init__(
                    self,
                    num_heads: int,
                    context_size: int,
                    d_model: int,
                    dim_keys: int,
                    dim_values: int,
                    drop_prob: float,
                    mask_attention: bool,
                    is_encoder: bool
                ):
        super(MultiHeadAttention, self).__init__()

        self.heads = nn.ModuleList([ScaledDotProductAttention   (  
                                                                context_size=context_size, 
                                                                d_model=d_model, 
                                                                dim_keys=dim_keys, 
                                                                dim_values=dim_values, 
                                                                drop_prob=drop_prob, 
                                                                mask_attention=mask_attention
                                                                ) for _ in range(num_heads)])

        self.linear = ResidualScaledLinear(num_heads * dim_values, d_model, is_encoder)
        
        self.dropout = nn.Dropout(drop_prob)

    def forward (
                    self, 
                    X_Q, 
                    X_KV, 
                    attention_mask: Optional[torch.FloatTensor] = None
                ):
        Ans = torch.cat([h(X_Q, X_KV, attention_mask) for h in self.heads], dim = -1)
        Ans = self.dropout(self.linear(Ans))

        return Ans

class FeedForward(nn.Module):
    def __init__(
                    self,
                    d_model: int,
                    d_ff: int,
                    drop_prob: float,
                    is_encoder: bool
                ):
        super(FeedForward, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        
        self.linear2 = ResidualScaledLinear(d_ff, d_model, is_encoder)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, X):
    
        return self.dropout(self.linear2(self.relu(self.linear1(X))))

class EncoderBlock(nn.Module):
    def __init__(
                    self,
                    num_heads: int,
                    context_size: int,
                    d_model: int,
                    drop_prob: float,
                    d_ff: int
                ):
        super(EncoderBlock, self).__init__()

        self.mha = MultiHeadAttention   (
                                            num_heads = num_heads, 
                                            context_size = context_size, 
                                            d_model = d_model, 
                                            dim_keys = d_model // num_heads, 
                                            dim_values = d_model // num_heads, 
                                            drop_prob = drop_prob, 
                                            mask_attention = False,
                                            is_encoder = True
                                        )

        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, drop_prob, is_encoder = True)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, X, pre_norm:bool):

        if pre_norm:
            # TODO: Below is the implementation of the Pre-Norm Formulation, Also remember to check 
            # original paper post norm formulation
            X_Norm = self.norm1(X)
            X = X + self.mha(X_Norm, X_Norm)
            X = X + self.ff(self.norm2(X))
        else:
            # According to original paper
            X = self.norm1(X + self.mha(X, X))
            X = self.norm2(X + self.ff(X))

        return X

class DecoderBlock(nn.Module):
    def __init__(
                    self,
                    num_heads: int,
                    context_size: int,
                    d_model: int,
                    drop_prob: float,
                    d_ff: int
                ):
        super(DecoderBlock, self).__init__()

        self.masked_mha = MultiHeadAttention   (
                                                    num_heads = num_heads, 
                                                    context_size = context_size, 
                                                    d_model = d_model, 
                                                    dim_keys = d_model // num_heads, 
                                                    dim_values = d_model // num_heads, 
                                                    drop_prob = drop_prob, 
                                                    mask_attention = True,
                                                    is_encoder = False
                                                )
                                                
        self.cross_mha = MultiHeadAttention   (
                                            num_heads = num_heads, 
                                            context_size = context_size, 
                                            d_model = d_model, 
                                            dim_keys = d_model // num_heads, 
                                            dim_values = d_model // num_heads, 
                                            drop_prob = drop_prob, 
                                            mask_attention = False,
                                            is_encoder = False
                                        )

        self.norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(4)])

        self.ff = FeedForward(d_model, d_ff, drop_prob, is_encoder = False)
    
    def forward(self, X, encoded_info, pre_norm:bool):
        
        if pre_norm:
            # TODO: Below is the implementation of the Pre-Norm Formulation, Also remember to check 
            # original paper post norm formulation
            X_Norm = self.norm[0](X)
            X = X + self.masked_mha(X_Norm, X_Norm)
            X = X + self.cross_mha(self.norm[1](X), self.norm[2](encoded_info))
            X = X + self.ff(self.norm[3](X))
        else:
            # According to original paper
            X = X + self.masked_mha(X, X)
            X = self.norm[0](X)
            X = X + self.cross_mha(X, encoded_info)
            X = self.norm[1](X)
            X = X + self.ff(X)
            X = self.norm[2](X)

        return X

class OnlyDecoderBlock(nn.Module):
    def __init__(
                    self,
                    num_heads: int,
                    context_size: int,
                    d_model: int,
                    drop_prob: float,
                    d_ff: int,
                    mask_attention: bool
                ):
        super(OnlyDecoderBlock, self).__init__()

        self.mha = MultiHeadAttention   (
                                                    num_heads = num_heads, 
                                                    context_size = context_size, 
                                                    d_model = d_model, 
                                                    dim_keys = d_model // num_heads, 
                                                    dim_values = d_model // num_heads, 
                                                    drop_prob = drop_prob, 
                                                    mask_attention = mask_attention,
                                                    is_encoder = False
                                                 )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = FeedForward(d_model, d_ff, drop_prob, is_encoder = False)
    
    def forward (
                    self, 
                    X, 
                    pre_norm:bool, 
                    attention_mask: Optional[torch.FloatTensor] = None
                ):
        
        if pre_norm:
            # TODO: Below is the implementation of the Pre-Norm Formulation, Also remember to check 
            # original paper post norm formulation
            X_Norm = self.norm1(X)
            X = X + self.mha(X_Norm, X_Norm, attention_mask)
            X = X + self.ff(self.norm2(X))
        else:
            # According to original paper
            X = X + self.masked_mha(X, X)
            X = self.norm1(X)
            X = X + self.ff(X)
            X = self.norm2(X)

        return X