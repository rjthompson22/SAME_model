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

