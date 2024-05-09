import torch
import torch.nn as nn
import unittest
from efficient_kan import KAN  # Importing KAN from the efficient_kan module

# MLP2 class definition using the imported KAN
class MLP2(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config['downsizeMLP']:
            kan_layers = [config['downsize_size'], config['kan_hidden_size'], config['downsize_size']]
        else:
            kan_layers = [config['n_embd'], config['kan_hidden_size'], config['n_embd']]

        self.kan = KAN(
            layers_hidden=kan_layers,
            grid_size=config['grid_size'],
            spline_order=config['spline_order'],
            scale_noise=config['scale_noise'],
            scale_base=config['scale_base'],
            scale_spline=config['scale_spline'],
            base_activation=config['base_activation'],
            grid_eps=config['grid_eps'],
            grid_range=config['grid_range'],
        )
        self.dropout = nn.Dropout(config['dropout'])
        self.downsizeMLP = config['downsizeMLP']
        if self.downsizeMLP:
            self.downsizer = nn.Sequential(nn.Linear(config['n_embd'], config['downsize_size']), nn.ReLU())
            self.upsizer = nn.Sequential(nn.Linear(config['downsize_size'], config['n_embd']), nn.ReLU())

    def forward(self, x):
        batch_size, seq_length, n_embd = x.shape
        x = x.view(batch_size * seq_length, n_embd)
        if self.downsizeMLP:
            x = self.downsizer(x)
        x = self.kan(x)
        if self.downsizeMLP:
            x = self.upsizer(x)
        x = x.view(batch_size, seq_length, n_embd)
        x = self.dropout(x)
        return x

# SimpleMLP class as described
class SimpleMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config['n_embd'], config['n_embd'], bias=config['bias'])
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config['n_embd'], config['n_embd'], bias=config['bias'])
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# Test suite to validate the MLP2 implementation and compare with SimpleMLP
class TestMLPModels(unittest.TestCase):
    def setUp(self):
        # Configuration for MLP testing
        self.config = {
            'n_embd': 768,
            'downsizeMLP': True,
            'downsize_size': 256,
            'kan_hidden_size': 512,
            'grid_size': 5,
            'spline_order': 3,
            'scale_noise': 0.1,
            'scale_base': 1.0,
            'scale_spline': 1.0,
            'base_activation': nn.SiLU,
            'grid_eps': 0.02,
            'grid_range': [-1, 1],
            'dropout': 0.1,
            'bias': True
        }
        self.mlp2 = MLP2(self.config)
        self.simple_mlp = SimpleMLP(self.config)

    def test_output_shape(self):
        # Create a dummy input tensor of shape [batch_size, seq_length, n_embd]
        batch_size, seq_length, n_embd = 2, 10, self.config['n_embd']
        dummy_input = torch.rand(batch_size, seq_length, n_embd)

        # Get outputs
        output_mlp2 = self.mlp2(dummy_input)
        output_simple_mlp = self.simple_mlp(dummy_input)

        # Check if output shapes match
        self.assertEqual(output_mlp2.shape, output_simple_mlp.shape)

if __name__ == '__main__':
    unittest.main()
