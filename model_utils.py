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


def patchify(batch, img_size, patch_size):
    """
    This code defines a function patchify that takes in a batch of images, the image size, and the patch size as inputs. It splits each image in the batch into patches of the specified patch size and returns a tensor containing all the flattened patches.

    The function starts by calculating the number of steps needed to cover the entire image by dividing the image size by the patch size.
    
    Then, it initializes an empty list called output to store the flattened patches.
    
    The nested for loops iterate over each patch in the image. The loop variables i and j represent the current patch's position.
    
    Inside the loop, the code calculates the coordinates of the current patch's top left, top right, bottom left, and bottom right corners.
    
    Using these coordinates, it extracts the patch from the batch by slicing the tensor batch along the dimensions corresponding to the patch's coordinates.
    
    The patch tensor is then flattened using nn.Flatten(1) to collapse all dimensions except the batch dimension. An additional dimension is added by unsqueezing the tensor along the first dimension.
    
    The flattened patch is appended to the output list.
    
    After processing all the patches, the code concatenates all the flattened patches along the second dimension using torch.cat.
    
    Finally, the concatenated tensor is returned as the output of the patchify function.
    """
    # Calculate the number of steps needed to cover the entire image
    step = int(img_size / patch_size)

    output = []
    for i in range(step):
        for j in range(step):
            # Calculate the coordinates of the current patch
            t_l = i * patch_size  # Top left corner
            t_r = (i + 1) * patch_size  # Top right corner
            b_l = j * patch_size  # Bottom left corner
            b_r = (j + 1) * patch_size  # Bottom right corner

            # Extract the patch from the batch
            patch = batch[:, :, t_l:t_r, b_l:b_r]

            # Flatten the patch and add an extra dimension
            flattened_patch = nn.Flatten(1)(patch).unsqueeze(1)

            # Add the flattened patch to the output list
            output += [flattened_patch]

    # Concatenate all the flattened patches along the second dimension
    output = torch.cat(output, dim=1)

    return output


def unpatchify(patches, img_size, patch_size, channels):
    """
    This code defines a function unpatchify that takes in a tensor of patches, the image size, patch size, and the number of channels as inputs. It reconstructs the original image from the patches and returns it.

    The function begins by calculating the number of steps needed to cover the entire image by dividing the image size by the patch size. It also obtains the batch size from the patches tensor.
    
    An empty list called patch_list is initialized to store the reshaped patches.
    
    The loop iterates over the second-to-last dimension of the patches tensor. This dimension corresponds to the patches' indices.
    
    Within the loop, each patch is reshaped to its original dimensions using the reshape function. The reshaped patch is added to the patch_list.
    
    After processing all the patches, the code constructs the rows of the image. It uses another loop to iterate over the range of steps.
    
    Within each iteration, a sublist of patches corresponding to a row is extracted from the patch_list using slicing. The sublists are then concatenated along the last dimension (-1) using torch.cat.
    
    The resulting rows of patches are stored in the img_rows list.
    
    Finally, the rows of patches are concatenated along the second-to-last dimension (-2) using torch.cat to reconstruct the original image. The reconstructed image is returned as the output of the unpatchify function.
    """
    # Calculate the number of steps needed to cover the entire image
    step = int(img_size / patch_size)
    batch_size = patches.shape[0]

    patch_list = []
    for i in range(patches.shape[-2]):
        # Reshape each patch to its original dimensions
        patch_list += [patches[:, i].reshape(batch_size, channels, patch_size, patch_size)]

    img_rows = [torch.cat(patch_list[i * step:(i * step) + step], -1) for i in range(step)]

    # Concatenate all the rows of patches to reconstruct the image
    img = torch.cat(img_rows, -2)

    return img
