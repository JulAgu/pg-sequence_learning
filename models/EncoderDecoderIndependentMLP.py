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
    
class DynamicEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 first_decoder_input_dim,
                 num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim,
                          hidden_dim,
                          num_layers=num_layers, 
                          batch_first=True)

        self.mlp_to_first_decoder_input = nn.Sequential(
            nn.Linear(input_dim, first_decoder_input_dim),
            nn.ReLU(),
            nn.Linear(first_decoder_input_dim, first_decoder_input_dim)
            )
    
    def forward(self, x):
        _, h_n = self.gru(input=x)
        first_decoder_input = self.mlp_to_first_decoder_input(x[:, -1, :])
        
        return h_n, first_decoder_input

class Encoder(nn.Module):
    def __init__(self,
                 static_input_dim,
                 static_hidden_dim,
                 list_unic_cat,
                 embedding_dims,
                 dynamic_input_dim,               
                 dynamic_hidden_dim,
                 first_decoder_input_dim,
                 gru_num_layers=2):
        super().__init__()
        self.static_encoder = StaticEncoder(static_input_dim,
                                            list_unic_cat,
                                            embedding_dims,
                                            static_hidden_dim)
        self.dynamic_encoder = DynamicEncoder(dynamic_input_dim,
                                              dynamic_hidden_dim,
                                              first_decoder_input_dim,
                                              gru_num_layers)

    def forward(self, static_features, static_cat_features, dynamic_features):
        static_encoded = self.static_encoder(static_features, static_cat_features)
        dynamic_encoded, first_decoder_input = self.dynamic_encoder(dynamic_features)
        latent = torch.cat([static_encoded.repeat(self.dynamic_encoder.gru.num_layers, 1, 1), dynamic_encoded], dim=2)
        return latent, first_decoder_input


class Decoder(nn.Module):
    def __init__(self,
                 gru_input_dim,
                 gru_hidden_dim,
                 stepwise_input_dim,
                 main_hidden_dim,
                 output_dim,
                 monotonic_indices=None,  # list of MLP indices that should be monotonic
                 num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_size=gru_input_dim,
                          hidden_size=gru_hidden_dim,
                          num_layers=num_layers,
                          batch_first=True)
        
        # Build MLPs dynamically
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(gru_hidden_dim + stepwise_input_dim, main_hidden_dim),
                nn.ReLU(),
                nn.Linear(main_hidden_dim, int(round(main_hidden_dim * 0.5))),
                nn.ReLU(),
                nn.Linear(int(round(main_hidden_dim * 0.5)), 1)
            )
            for _ in range(output_dim)
        ])

    def forward(self, x_0, h_0, stepwise_input):
        _, h = self.gru(x_0, h_0)
        h_augmented = torch.cat([h[-1], stepwise_input], dim=1)

        outputs = [mlp(h_augmented) for mlp in self.mlps]
        output = torch.cat(outputs, dim=1)

        return output, h, h_augmented

class StopPredictor(nn.Module):
    def __init__(self,
                 gru_hidden_dim,
                 stepwise_input_dim,
                 mask_hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(gru_hidden_dim + stepwise_input_dim, mask_hidden_dim),
                                         nn.ReLU(),
                                         nn.Linear(mask_hidden_dim, int(round(mask_hidden_dim * 0.5))),
                                         nn.ReLU(),
                                         nn.Linear(int(round(mask_hidden_dim * 0.5)), 1))

    def forward(self, h_augmented):
        output = self.fc(h_augmented)
        return output