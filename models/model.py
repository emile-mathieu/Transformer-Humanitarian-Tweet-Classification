import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_length):
        super(PositionalEncoding, self).__init__()

        # Create positional encoding matrix
        positional_encoding = torch.zeros(1, max_seq_length, embed_dim)

        position = torch.arange(0, max_seq_length).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))

        # Even dimensions use sin
        positional_encoding[0, :, 0::2] = torch.sin(position * div_term)

        # Odd dimensions use cos
        positional_encoding[0, :, 1::2] = torch.cos(position * div_term[:embed_dim // 2])

        # Positional encoding is fixed, not learnable
        self.register_buffer(
            "positional_encoding",
            positional_encoding
        )

    def forward(self, x):
        return x + self.positional_encoding[:, :x.size(1), :]


class TweetTransformer(nn.Module):
    def __init__(
        self,
        vocab_size=16000,
        embed_dim=512,
        dim_feedforward=2048,
        num_heads=8,
        num_layers=6,
        num_classes=11,
        max_seq_length=128,
        dropout=0.1,
    ):
        super(TweetTransformer, self).__init__()
        # Token embeddings
        self.embedding = nn.Embedding(vocab_size,embed_dim)
        # Sinusoidal positional encoding
        
        self.positional_encoding = PositionalEncoding(embed_dim,max_seq_length)
        # Single Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )

        # Stack encoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification layer
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        # Get token embeddings
        embeddings_input = self.embedding(input_ids)
        # Add positional encoding
        pos_embeddings_input = self.positional_encoding(embeddings_input)

        # In PyTorch's Transformer, attention mask is = 1 for positions we want to attend to and 0 for masked positions. 
        # We need to invert it.

        padding_mask = attention_mask == 0  # True for positions to be masked
        encoder_output = self.transformer_encoder(pos_embeddings_input, src_key_padding_mask=padding_mask)

        # Use the output corresponding to the first token "SOS" for classification
        prediction = self.fc(encoder_output[:, 0, :])
        return prediction