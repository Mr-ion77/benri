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
        entangling_layers = 0, 
        train_q = False # Now controls ONLY entangling params
    ):
        super().__init__()

        self.num_qubits = num_qubits
        self.entangle_method = entangle_method
        self.U3_layers = int(U3_layers)
        self.entangling_layers = int(entangling_layers)
        self.invert = invert
        self.train_q = train_q

        self.anytrain = self.train_q or self.U3_layers > 0
    
        print(f"Initialized QuantumLayer with {self.U3_layers} U3 layers and {self.entangling_layers} entangling layers. Trainable entangling params: {self.train_q}. Any trainable params: {self.anytrain}.")

        # 1. Graph Processing (Logic remains same)
        if not isinstance(graphs, list): graphs = [graphs]
        if len(graphs) < self.entangling_layers:
            new_graphs = [graphs[i % len(graphs)] for i in range(self.entangling_layers)]
            graphs = new_graphs

        self.layers_edges = []
        self.layers_weights = []
        for g in graphs[:self.entangling_layers]:
            data = graph_builder(g, num_qubits) if isinstance(g, str) else g
            self.layers_edges.append(data['edges'])
            self.layers_weights.append(data['weights'])

        # 2. Parameter Indexing
        # We split weights into two distinct groups for TorchLayer to manage
        self.u3_param_count = 3 * self.num_qubits * self.U3_layers
        
        self.entangle_param_count = 0
        if self.entangle_method in ['CRX', 'CRY']:
            for edges in self.layers_edges:
                self.entangle_param_count += len(edges)
        
        # 3. Device & Circuit
        device_name = 'default.qubit' #if self.anytrain else 'lightning.qubit'
        dev = qml.device(device_name, wires=num_qubits)
        
        _diff_method = "backprop" #if self.anytrain else None

        print(f"Initializing circuit on device '{device_name}' with {self.num_qubits} qubits. Diff method: {_diff_method}. U3 params: {self.u3_param_count}, Entangling params: {self.entangle_param_count}.")

        @qml.qnode(dev, interface="torch", diff_method=_diff_method)
        def circuit_(inputs, u3_weights, ent_weights):
            # Input Encoding
            inputs = np.pi * (1 - self.invert + (2 * self.invert - 1) * torch.clamp(inputs, min=0, max=1))
            qml.AngleEmbedding(inputs, wires=range(self.num_qubits), rotation='Y')
            
            u3_idx = 0
            ent_idx = 0
            max_layers = max(self.U3_layers, self.entangling_layers)
            
            for L in range(max_layers):
                # Rotation Layer (U3) - Always trainable if present
                if L < self.U3_layers:
                    for q in range(self.num_qubits):
                        qml.Rot(*u3_weights[u3_idx:u3_idx+3], wires=q)
                        u3_idx += 3
                
                # Entanglement Layer - Frozen based on train_q
                if L < self.entangling_layers:
                    current_edges = self.layers_edges[L]
                    current_fixed_weights = self.layers_weights[L]

                    for i, (u, v) in enumerate(current_edges):
                        if self.entangle_method == 'CNOT':
                            qml.CNOT(wires=[u, v])
                        else:
                            w = ent_weights[ent_idx]
                            ent_idx += 1
                            if self.entangle_method == 'CRX':
                                qml.CRX(w + current_fixed_weights[i], wires=[u, v])
                            elif self.entangle_method == 'CRY':
                                qml.CRY(w + current_fixed_weights[i], wires=[u, v])

            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        # 4. TorchLayer Setup with weight separation
        weight_shapes = {
            "u3_weights": (self.u3_param_count,) if self.u3_param_count > 0 else (0,),
            "ent_weights": (self.entangle_param_count,) if self.entangle_param_count > 0 else (0,)
        }

        self.circuit_ = circuit_
        self.magic = qml.qnn.TorchLayer(circuit_, weight_shapes)
        
        # 5. Selective Freezing
        if not self.train_q:
            if hasattr(self.magic, 'ent_weights'):
                self.magic.ent_weights.requires_grad = False
                # Optionally zero them out if that was your goal
                with torch.no_grad():
                    self.magic.ent_weights.fill_(0)

    def forward(self, inputs):
        
        if self.anytrain:
             
            if inputs.ndim == 3:
                return torch.vmap(torch.vmap(self.magic))(inputs)
            elif inputs.ndim == 2:
                return torch.vmap(self.magic)(inputs)
            
        else:
            # INFERENCE: Bypass TorchLayer, let lightning.qubit handle the batch in C++
            # We must pass dummy weights since TorchLayer isn't managing them here
            dummy_u3 = torch.empty(0)
            dummy_ent = self.magic.ent_weights if hasattr(self.magic, 'ent_weights') else torch.empty(0)
            
            return self.circuit_(inputs, dummy_u3, dummy_ent)