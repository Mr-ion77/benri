import torch
import pennylane as qml
import numpy as np
from .graphs import graph_builder
import warnings

class QuantumLayer(torch.nn.Module):
    """
        A trainable quantum neural network layer integrated with PyTorch.

        This layer implements a variational quantum circuit using PennyLane. It supports 
        flexible entanglement structures based on graph topologies, U3 rotation layers, 
        and multiple entangling methods. It is designed to handle batched inputs 
        automatically via torch.vmap.

        Attributes:
            num_qubits (int): Number of wires in the quantum device.
            entangle_method (str): The type of entangling gate used ('CNOT', 'CRX', 'CRY', or 'SEL' -Strongly Entangling Layers-).
            U3_layers (int): Number of trainable U3 rotation layers to apply.
            entangling_layers (int): Number of entangling layers based on provided graphs.
            invert (bool): If True, scales inputs using (1 - inputs), otherwise uses raw inputs.
            layers_edges (list): List of edge lists for each entangling layer.
            layers_weights (list): List of fixed weights associated with the graph edges.
            magic (qml.qnn.TorchLayer): The PennyLane-PyTorch interface layer.

        Args:
            num_qubits (int): Total qubits/wires for the circuit.
            graphs (str, dict, or list): Graph specification(s) for entanglement. 
                Can be a single string/dict or a list of them. If the list is shorter 
                than `entangling_layers`, it cycles through the available graphs.
            entangle_method (str, optional): Gate type for entanglement. 
                Defaults to 'CNOT'. Options: 'CNOT', 'CRX', 'CRY', 'SEL'.
            invert (bool, optional): Input normalization toggle. Defaults to True.
            U3_layers (int, optional): Number of layers of U3 gates. Defaults to 0.
            entangling_layers (int, optional): Number of layers of entanglement. Defaults to 0.

        Raises:
            ValueError: If a graph type is not a string or dictionary, or if input 
                tensor dimensions in `forward` are not 1D, 2D, or 3D.
    """
    def __init__(
        self,
        num_qubits,
        graphs, 
        entangle_method='CNOT',
        invert=True, 
        U3_layers = 0,
        entangling_layers = 0
    ):
        super().__init__()
        self.num_qubits = num_qubits
        self.entangle_method = entangle_method
        self.U3_layers = int(U3_layers)
        self.entangling_layers = int(entangling_layers)
        self.invert = invert
        
        # --- 1. Multi-Graph Processing with Cycling ---
        if not isinstance(graphs, list):
            graphs = [graphs]
        
        # If the provided list is too short, warn the user and cycle the list
        if len(graphs) < self.entangling_layers:
            print(f"Warning: Provided {len(graphs)} graphs for {self.entangling_layers} entangling layers. "
                  f"Cycling the provided graphs to fill the layers.")
            
            # Construct a new list by cycling the original list
            new_graphs = []
            for i in range(self.entangling_layers):
                new_graphs.append(graphs[i % len(graphs)])
            graphs = new_graphs

        self.layers_edges = []
        self.layers_weights = []

        # We take exactly the number of graphs needed for the entangling layers
        for g in graphs[:self.entangling_layers]:
            if isinstance(g, str):
                data = graph_builder(g, num_qubits)
            elif isinstance(g, dict):
                data = g
            else:
                raise ValueError(f"Invalid graph type: {type(g)}")
            
            self.layers_edges.append(data['edges'])
            self.layers_weights.append(data['weights'])

        # --- 2. Weight Shape Calculation ---
        u3_params = 3 * self.num_qubits * self.U3_layers
        
        entangle_params = 0
        if self.entangle_method in ['CRX', 'CRY']:
            for edges in self.layers_edges:
                entangle_params += len(edges)
            
        total_weights = u3_params + entangle_params
        weight_shape = (total_weights,) if total_weights > 0 else (1,)

        # --- 3. Device & Circuit ---
        dev = qml.device('default.qubit', wires=num_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit_(inputs, weights):
            inputs = np.pi * (1 - self.invert + (2 * self.invert - 1) * torch.clamp(inputs, min=0, max=1))
            
            
            qml.AngleEmbedding(inputs, wires=range(self.num_qubits), rotation='Y')

            if entangle_method != 'SEL':

                w_idx = 0
                max_layers = max(self.U3_layers, self.entangling_layers)
                
                for L in range(max_layers):
                    # Rotation Layer (U3)
                    if L < self.U3_layers:
                        for q in range(self.num_qubits):
                            qml.Rot(weights[w_idx], weights[w_idx+1], weights[w_idx+2], wires=q)
                            w_idx += 3
                    
                    # Entanglement Layer
                    if L < self.entangling_layers:
                        current_edges = self.layers_edges[L]
                        current_fixed_weights = self.layers_weights[L]

                        for i, (u, v) in enumerate(current_edges):
                            if self.entangle_method in ['CRX', 'CRY']:
                                w = weights[w_idx]
                                w_idx += 1
                            else:
                                w = current_fixed_weights[i] if i < len(current_fixed_weights) else (np.pi / 3)

                            if self.entangle_method == 'CNOT':
                                qml.CNOT(wires=[u, v])
                            elif self.entangle_method == 'CRX':
                                qml.CRX(w, wires=[u, v])
                            elif self.entangle_method == 'CRY':
                                qml.CRY(w, wires=[u, v])

            if self.entangle_method == 'SEL':
                qml.StronglyEntanglingLayers(weights.reshape(1, self.num_qubits, 3), wires=range(self.num_qubits))

            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        # --- 4. Setup ---
        self.magic = qml.qnn.TorchLayer(circuit_, {"weights": weight_shape})
        
        if total_weights == 0:
            for param in self.magic.parameters():
                param.requires_grad = False
        
        print(f"Quantum Layer Info: U3 Layers={self.U3_layers} | Entangling Layers={self.entangling_layers}")
        print(f"Graphs utilized: {len(self.layers_edges)} | Total Params: {total_weights}")

    def forward(self, inputs):
        if inputs.ndim == 3:
            return torch.vmap(torch.vmap(self.magic))(inputs)
        elif inputs.ndim == 2:
            return torch.vmap(self.magic)(inputs)
        elif inputs.ndim == 1:
            return self.magic(inputs)
        else:
            raise ValueError("Input tensor must be 1D, 2D or 3D")