import torch
from torch import nn

BATCH_SIZE = 32
PATCH_SIZE = 16
IMAGE_WIDTH = 224
IMAGE_HEIGHT = IMAGE_WIDTH
IMAGE_CHANNELS = 3
EMBEDDING_DIMS = IMAGE_CHANNELS * PATCH_SIZE ** 2
NUM_OF_PATCHES = int((IMAGE_WIDTH * IMAGE_HEIGHT) / PATCH_SIZE ** 2)

# the image width and image height should be divisible by patch size. This is a check to see that.

assert IMAGE_WIDTH % PATCH_SIZE == 0 and IMAGE_HEIGHT % PATCH_SIZE == 0, print(
    "Image Width is not divisible by patch size"
)

class PatchEmbeddingLayer(nn.Module):
    def __init__(self, in_channels, patch_size, embedding_dim):
        super().__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.in_channels = in_channels
        self.conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.flatten_layer = nn.Flatten(start_dim=1, end_dim=2)
        self.class_token_embeddings = nn.Parameter(
            torch.rand((BATCH_SIZE, 1, EMBEDDING_DIMS), requires_grad=True)
        )
        self.position_embeddings = nn.Parameter(
            torch.rand((1, NUM_OF_PATCHES + 1, EMBEDDING_DIMS), requires_grad=True)
        )
    def forward(self, x):
        output = (
                torch.cat(
                    (
                        self.class_token_embeddings,
                        self.flatten_layer(self.conv_layer(x).permute((0, 2, 3, 1))),
                    ),
                    dim=1,
                )
                + self.position_embeddings
        )
        return output

class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self,
                 embedding_dims=768,  # Hidden Size D in the ViT Paper Table 1
                 num_heads=12,  # Heads in the ViT Paper Table 1
                 attn_dropout=0.0  # Default to Zero as there is no dropout for the the MSA Block as per the ViT Paper
                 ):
        super().__init__()

        self.embedding_dims = embedding_dims
        self.num_head = num_heads
        self.attn_dropout = attn_dropout

        self.layernorm = nn.LayerNorm(normalized_shape=embedding_dims)

        self.multiheadattention = nn.MultiheadAttention(num_heads=num_heads,
                                                        embed_dim=embedding_dims,
                                                        dropout=attn_dropout,
                                                        batch_first=True,
                                                        )
    def forward(self, x):
        x = self.layernorm(x)
        output, _ = self.multiheadattention(query=x, key=x, value=x, need_weights=False)
        return output

class MultiLayerPerceptronBlock(nn.Module):
    def __init__(self, embedding_dims, mlp_size, mlp_dropout):
        super().__init__()
        self.embedding_dims = embedding_dims
        self.mlp_size = mlp_size
        self.dropout = mlp_dropout

        self.layernorm = nn.LayerNorm(normalized_shape=embedding_dims)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dims, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=mlp_dropout),
            nn.Linear(in_features=mlp_size, out_features=embedding_dims),
            nn.Dropout(p=mlp_dropout)
        )
    def forward(self, x):
        return self.mlp(self.layernorm(x))

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dims=768,
                 mlp_dropout=0.1,
                 attn_dropout=0.0,
                 mlp_size=3072,
                 num_heads=12,
                 ):
        super().__init__()

        self.msa_block = MultiHeadSelfAttentionBlock(embedding_dims=embedding_dims,
                                                     num_heads=num_heads,
                                                     attn_dropout=attn_dropout)

        self.mlp_block = MultiLayerPerceptronBlock(embedding_dims=embedding_dims,
                                                   mlp_size=mlp_size,
                                                   mlp_dropout=mlp_dropout,
                                                   )
    def forward(self, x):
        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x

        return x

class ViT(nn.Module):
    def __init__(self, img_size=224,
                 in_channels=3,
                 patch_size=16,
                 embedding_dims=768,
                 num_transformer_layers=12,  # from table 1 above
                 mlp_dropout=0.1,
                 attn_dropout=0.0,
                 mlp_size=3072,
                 num_heads=12,
                 num_classes=1000):
        super().__init__()

        self.patch_embedding_layer = PatchEmbeddingLayer(in_channels=in_channels,
                                                         patch_size=patch_size,
                                                         embedding_dim=embedding_dims)

        self.transformer_encoder = nn.Sequential(*[TransformerBlock(embedding_dims=embedding_dims,
                                                                    mlp_dropout=mlp_dropout,
                                                                    attn_dropout=attn_dropout,
                                                                    mlp_size=mlp_size,
                                                                    num_heads=num_heads) for _ in
                                                   range(num_transformer_layers)])

        self.classifier = nn.Sequential(nn.LayerNorm(normalized_shape=embedding_dims),
                                        nn.Linear(in_features=embedding_dims,
                                                  out_features=num_classes))
    def forward(self, x):
        return self.classifier(self.transformer_encoder(self.patch_embedding_layer(x))[:, 0])
