import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.data import Data

__all__ = ['GNNTracker']

class GNNTracker(nn.Module):
	"""Graph neural network node classifier.

	based on : https://github.com/pyg-team/pytorch_geometric/blob/master/examples/ogbn_proteins_deepgcn.py

	Attributes
	----------
	edge_encoder :
		edge feature encoder. maps edge_dim -> hidden_channels
	node_encoder :
		node feature encoder. maps node_dim -> hidden_channels
	layers :
		list of DeepGCNLayer. applies graph convolution
	lin :
		final linear layer. maps hidden_channels -> 1
	"""

	def __init__(self,
		num_node_attr: int,
		num_edge_attr: int,
		dropout_rate: float,
		encoder_hidden_channels: int,
		encoder_num_layers: int,
		conv_hidden_channels: int,
		conv_num_layers: int,
		num_classes: int,
	):
		super().__init__()

		self.num_node_attr = num_node_attr
		self.num_edge_attr = num_edge_attr
		self.dropout_rate = dropout_rate
		self.encoder_hidden_channels = encoder_hidden_channels
		self.encoder_num_layers = encoder_num_layers
		self.conv_hidden_channels = conv_hidden_channels
		self.conv_num_layers = conv_num_layers
		self.num_classes = num_classes

		self.node_encoder = gnn.MLP(
			in_channels=self.num_node_attr,
			hidden_channels=self.encoder_hidden_channels,
			out_channels=self.conv_hidden_channels,
			num_layers=self.encoder_num_layers,
			dropout=self.dropout_rate,
			norm='batch_norm',
		)
		self.edge_encoder = gnn.MLP(
			in_channels=self.num_edge_attr,
			hidden_channels=self.encoder_hidden_channels,
			out_channels=self.conv_hidden_channels,
			num_layers=self.encoder_num_layers,
			dropout=self.dropout_rate,
			norm='batch_norm',
		)

		self.layers = nn.ModuleList()

		for i in range(1, self.conv_num_layers + 1):
			conv = gnn.GENConv(
				self.conv_hidden_channels,
				self.conv_hidden_channels,
				aggr='softmax',
				t=1.0, learn_t=True, num_layers=2, norm='layer'
			)
			norm = nn.LayerNorm(self.conv_hidden_channels, elementwise_affine=True)
			act = nn.ReLU(inplace=True)

			layer = gnn.DeepGCNLayer(
				conv, norm, act,
				block='res+', dropout=self.dropout_rate, ckpt_grad=i % 3)

			self.layers.append(layer)

		self.out = gnn.MLP(
			[self.conv_hidden_channels, self.conv_hidden_channels//2, self.conv_hidden_channels//4, self.num_classes],
			dropout=0.0,
			norm='batch_norm',
		)

	def forward(self, graph: Data):
		x = self.node_encoder(graph.x)
		edge_attr = self.edge_encoder(graph.edge_attr)
		edge_index = graph.edge_index

		x = self.layers[0].conv(x, edge_index, edge_attr)

		for layer in self.layers[1:]:
			x = layer(x, edge_index, edge_attr)

		return self.out(x)