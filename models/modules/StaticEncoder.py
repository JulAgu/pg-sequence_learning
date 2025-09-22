import torch
import torch.nn as nn

class StaticEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 list_unic_cat,
                 embedding_dims,
                 hidden_dim):
        super().__init__()

        self.embeddings = nn.ModuleList(
                    [
                        nn.Embedding(num_embeddings=i, embedding_dim=dimension)
                        for i, dimension in zip(list_unic_cat, embedding_dims)
                    ]
                )
        self.mlp_after_embedding = nn.Sequential(
            nn.Linear(sum(embedding_dims), int(round(sum(embedding_dims)*0.7))),
            nn.ReLU(),
            nn.Linear(int(round(sum(embedding_dims)*0.7)), hidden_dim))

        self.fc = nn.Sequential(
            nn.Linear(input_dim-len(list_unic_cat), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.last_ffnn = nn.Linear(hidden_dim*2, hidden_dim)
    def forward(self, x_num, x_cat):
        emb = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        emb = torch.cat(emb, dim=1)
        emb = self.mlp_after_embedding(emb)
        x_num = self.fc(x_num)
        x = torch.cat([x_num, emb], dim=1)
        x_prime = self.last_ffnn(x)
        return x_prime