import torch
import torch.nn as nn

class HybridArchiSimplified(nn.Module):
    def __init__(self,
                 input_dim,
                 input_dim_gru,
                 list_unic_cat,
                 embedding_dims,
                 num_gru_layers,
                 hidden_dim_gru,
                 hidden_dim,
                 output_dim):
        super().__init__()

        self.embeddings = nn.ModuleList(
                    [
                        nn.Embedding(num_embeddings=i, embedding_dim=dimension)
                        for i, dimension in zip(list_unic_cat, embedding_dims)
                    ]
                )
        self.fc_after_embedding = nn.Sequential(
            nn.Linear(sum(embedding_dims), int(round(sum(embedding_dims)*0.7))),
            nn.ReLU(),
            nn.Linear(int(round(sum(embedding_dims)*0.7)), hidden_dim))

        self.mlp_numerical = nn.Sequential(
            nn.Linear(input_dim-len(list_unic_cat), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.gru = nn.GRU(input_dim_gru,
                          hidden_dim_gru,
                          num_layers=num_gru_layers, 
                          batch_first=True)

        self._after_gru = nn.Sequential(
            nn.Linear(hidden_dim_gru, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        
        self.last_mlp = nn.Sequential(
            nn.Linear(hidden_dim*3, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim))

    def forward(self, input_num, input_cat, input_dyn):
        emb = [emb(input_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        emb = torch.cat(emb, dim=1)
        z_emb = self.fc_after_embedding(emb)
        z_num = self.mlp_numerical(input_num)
        _, h_n = self.gru(input=input_dyn)
        z_dyn = self._after_gru(h_n[-1])
        x = torch.cat([z_num, z_emb, z_dyn], dim=1)
        output = self.last_mlp(x)
        return output.squeeze(dim=1)