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
        
        if monotonic_indices is None:
            monotonic_indices = []
        self.monotonic_indices = set(monotonic_indices)  # make lookup fast

        # Build n MLPs dynamically
        self.mlps = nn.ModuleList()
        for i in range(output_dim):
            layers = [
                nn.Linear(gru_hidden_dim + stepwise_input_dim, main_hidden_dim),
                nn.ReLU(),
                nn.Linear(main_hidden_dim, int(round(main_hidden_dim * 0.5))),
                nn.ReLU(),
                nn.Linear(int(round(main_hidden_dim * 0.5)), 1)
            ]
            # Add final ReLU if monotonic
            if i in self.monotonic_indices:
                layers.append(nn.ReLU())
            self.mlps.append(nn.Sequential(*layers))

    def _forward_tf(self,
                    x,
                    h_0,
                    stepwise_input):
        """
        Notice that this is exactly equivalent to do teacher forcing with all time steps at once.
        """
        gru_out, h = self.gru(x,
                              h_0)
        # x : (batch_size, seq_len, input_dim)
        # gru_out: (batch_size, seq_len, gru_hidden_dim)

        h_augmented = torch.cat([gru_out, stepwise_input],
                                dim=2)
        
        # h_augmented: (batch_size, seq_len, gru_hidden_dim + stepwise_input_dim)

        outputs = []
        for i, mlp in enumerate(self.mlps):
            if i in self.monotonic_indices:
                outputs.append(mlp(h_augmented) + x[:, :, i].unsqueeze(2))
            # mlp(h_augmented): (batch_size, seq_len, 1)
            else:
                outputs.append(mlp(h_augmented))

        output = torch.cat(outputs, dim=2)
        return output, h_augmented

    def _forward_ar(self,
                    x_0,
                    h_0,
                    stepwise_input):
        _, h = self.gru(x_0, h_0)
        h_augmented = torch.cat([h[-1], stepwise_input], dim=1)

        outputs = []
        for i, mlp in enumerate(self.mlps):
            if i in self.monotonic_indices:
                outputs.append(mlp(h_augmented) + x_0[:, :, i])
            else:
                outputs.append(mlp(h_augmented))

        output = torch.cat(outputs, dim=1)

        return output, h, h_augmented

    def forward(self,
                x,
                h,
                stepwise_input,
                ar=False):
        if ar:
            return self._forward_ar(x_0=x,
                                    h_0=h,
                                    stepwise_input=stepwise_input)
        else:
            return self._forward_tf(x=x,
                                    h_0=h,
                                    stepwise_input=stepwise_input)


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