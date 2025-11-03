import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self,
                 input_dim,
                 list_unic_cat,
                 embedding_dims,
                 hidden_dim,
                 output_dim):
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
        
        self.numerical_size = input_dim - len(list_unic_cat)

        self.mlp_numerical = nn.Sequential(
            nn.Linear(self.numerical_size, round(self.numerical_size*0.9)),
            nn.ReLU(),
            nn.Linear(round(self.numerical_size*0.9), round(self.numerical_size*0.7)),
            nn.ReLU(),
            nn.Linear(round(self.numerical_size*0.7), round(self.numerical_size*0.7)),
            nn.ReLU(),
            nn.Linear(round(self.numerical_size*0.7), round(self.numerical_size*0.5)),
            nn.ReLU(),
            nn.Linear(round(self.numerical_size*0.5), round(self.numerical_size*0.5)),
            nn.ReLU(),
            nn.Linear(round(self.numerical_size*0.5), round(self.numerical_size*0.5)),
            nn.ReLU(),
            nn.Linear(round(self.numerical_size*0.5), round(self.numerical_size*0.3)),
            nn.ReLU(),
            nn.Linear(round(self.numerical_size*0.3), hidden_dim),
        )

        self.last_mlp = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, round(hidden_dim*0.5)),
            nn.ReLU(),
            nn.Linear(round(hidden_dim*0.5), output_dim)
        )

    def forward(self, x_num, x_cat):
        emb = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        emb = torch.cat(emb, dim=1)
        X_emb = self.mlp_after_embedding(emb)
        x_num = self.mlp_numerical(x_num)
        x = torch.cat([x_num, X_emb], dim=1)
        output = self.last_mlp(x)
        
        return output.squeeze(dim=1)  # (B,)