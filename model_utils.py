import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def PositionalEncoding(sequence_len, output_dim, n=10000):
    # Initialize the position encoding matrix with zeros
    P = torch.zeros((sequence_len, output_dim))

    # Iterate through the positions in the sequence
    for k in range(sequence_len):
        # Iterate through the dimensions of the encoding
        for i in range(0, output_dim, 2):  # Increment by 2 to handle both sine and cosine parts
            denominator = torch.tensor(n, dtype=torch.float).pow(2 * i / output_dim)
            P[k, i] = torch.sin(k / denominator)
            P[k, i + 1] = torch.cos(k / denominator)
            
    return P

def test_PositionalEncoding():
    # Get position encoding for a larger input
    pos_enc = PositionalEncoding(sequence_len=256, output_dim=256, n=1000)

    # Plot the position encoding matrix
    plt.rcParams["figure.figsize"] = (9, 9)
    pos_enc_graph = plt.matshow(pos_enc)
    plt.gcf().colorbar(pos_enc_graph)

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, device='cpu'):
        super().__init__()

        # Store the image size and patch size as instance variables
        self.img_size = img_size
        self.patch_size = patch_size

        # Calculate the total number of patches in the image
        self.n_patches = (img_size // patch_size) * (img_size // patch_size)

        # Define the patch embedding layer using a convolutional layer
        # The kernel size is set to the patch size and the stride is also set to the patch size
        # This results in non-overlapping patches of the image being processed
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, (embed_dim // 2), kernel_size=patch_size, stride=patch_size),
            nn.Flatten(1)
        )
        
        # Create positional encoding of the size to be concatenated with the patches
        self.pos_encoding = PositionalEncoding(sequence_len=self.n_patches, output_dim=(embed_dim // 2), n=1000).to(device)

        # Define the class token, which will be added to the sequence of patch embeddings
        # The class token is learnable and will be used to predict the class of the image
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x):
        # Get the batch size, channels, height, and width of the input image
        B, C, H, W = x.shape

        # Pass the image through the patch embedding layer
        x = self.patch_embedding(x)

        # Reshape the output of the patch embedding layer
        # Each row now represents a flattened patch of the image
        x = x.view(B, self.n_patches, -1)

        # Create a batch of class tokens by expanding the class token tensor
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        # Add the positional encoding to each patch
        x = torch.cat([self.pos_encoding.expand(B, -1, -1), x], dim=2)
        
        # Concatenate the class tokens with the patch embeddings
        x = torch.cat([cls_tokens, x], dim=1)

        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        # Store the embedding dimension and the number of attention heads
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Calculate the dimension of each head d
        self.head_dim = embed_dim // num_heads

        # Define the linear layer for projecting the input embeddings into query, key, and value tensors
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)

        # Define the attention dropout layer to prevent overfitting during training
        self.att_drop = nn.Dropout(0.1)

        # Define the linear layer for projecting the concatenated output of the attention heads
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # Get the batch size, sequence length, and embedding dimension
        B, N, E = x.shape

        # Pass the input through the query, key, and value projection layer
        qkv = self.qkv_proj(x)

        # Reshape the output tensor and separate query, key, and value tensors
        qkv = qkv.view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Calculate the attention scores by taking the dot product of the query and key tensors
        attn = (q @ k.transpose(-2, -1)) 
        
        # Use a scaling factor to prevent vanishing gradients
        attn *= (self.head_dim ** -0.5)

        # Apply the softmax function to the attention scores
        attn = nn.functional.softmax(attn, dim=-1)

        # Apply dropout to the attention scores
        attn = self.att_drop(attn)

        # Calculate the output by taking the dot product of the attention scores and value tensors
        x = (attn @ v).transpose(1, 2).contiguous().view(B, N, E)

        # Pass the output through the projection layer
        x = self.proj(x)

        return x
    
    
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super().__init__()

        # Define the Multi-Head Self-Attention layer with the given embedding dimension and number of heads
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)

        # Define the first layer normalization for the residual connection after the attention layer
        self.norm1 = nn.LayerNorm(embed_dim)

        # Define the MLP (feed-forward) layer, which consists of two linear layers and a GELU activation
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )

        # Define the second layer normalization for the residual connection after the MLP layer
        self.norm2 = nn.LayerNorm(embed_dim)

        # Define the dropout layer to prevent overfitting during training
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Pass the input through the Multi-Head Self-Attention layer
        attn_out = self.attention(x)

        # Add the attention output to the input (residual connection) and apply layer normalization
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # Pass the normalized output through the MLP layer
        mlp_out = self.mlp(x)

        # Add the MLP output to the input (residual connection) and apply layer normalization
        x = x + self.dropout(mlp_out)
        x = self.norm2(x)

        return x
    
class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_heads, mlp_dim, num_layers, num_classes, device):
        super().__init__()

        # Define the Patch Embedding layer, which divides the input image into patches and creates a sequence of embeddings
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim, device)

        # Define a sequence of Transformer Encoder layers
        self.transformer_encoders = nn.Sequential(
            *[TransformerEncoder(embed_dim, num_heads, mlp_dim) for _ in range(num_layers)]
        )

        # Define the class token, which is a learnable tensor that will be used to predict the class of the input image
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Define the final layer normalization to stabilize the output of the Transformer Encoder layers
        self.norm = nn.LayerNorm(embed_dim)

        # Define the classification head, which is a linear layer that maps the class token to the number of output classes
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Pass the input image through the Patch Embedding layer to create a sequence of embeddings
        x = self.patch_embed(x)

        # Pass the sequence of embeddings through the Transformer Encoder layers
        x = self.transformer_encoders(x)

        # Extract the class token from the output sequence
        cls_token = x[:, 0]

        # Apply layer normalization to the class token
        cls_token = self.norm(cls_token)

        # Pass the class token through the classification head to compute the logits for each class
        logits = self.head(cls_token)

        return logits