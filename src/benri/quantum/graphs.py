import numpy as np

def graph_builder(graph, num_qubits):
    """
    Builder function for predefined graphs.
    
    :param graph: Type of graph to build: 'chain' (entangle sequentially), 'chain_reverse' (reversed chain),  'neighbours' (chain + chain_reversed), 'star' ( entangle sequentially + 2-order-neighbours ).
    :param num_quibts: Number of qubits in the graph
    """

    special_graphs = {

        4 : {
                'X'                 :   [[0, 3], [1, 2], [3, 0], [2, 1]] 
        },

        6 : {   'david_star'        :   [[0, 2], [1, 3], [2, 4], [3, 5], [4, 0], [5, 1] ],
             
                'entangled_triangle':   [[0, 2], [1, 3], [2, 4], [3, 5], [4, 0], [5, 1], [0, 3]]
        },

        9 : {
                'king'              :   [[0, 2], [2, 8], [8, 6], [6, 0], [1, 5], [5, 7], [7, 3], [3, 1], [0, 4],
                                         [1, 4], [2, 4], [3, 4], [5, 4], [6, 4], [7, 4], [8, 4]],

                'center'            :   [[0, 4], [1, 4], [2, 4], [3, 4], [5, 4], [6, 4], [7, 4], [8, 4]]
        }
    }

    special_bool = graph in special_graphs.get(num_qubits, {})

    assert special_bool or graph in ['chain', 'star', 'chain_reverse', 'neighbours'], \
        f"Graph must be one fo the following: {list(special_graphs.keys()) + ['chain', 'star','chain_reverse','neighbours']}, but got {graph}."
    
    graph_edges = []

    if not special_bool:
    
        for i in range(num_qubits):
            if graph in ['chain', 'star', 'neighbours']:
                graph_edges.append( [ i, (i + 1) % num_qubits ] )
                if graph == 'star':
                    graph_edges.append( [ i , (i + 2) % num_qubits ])
                if graph == 'neighbours':
                    graph_edges.append( [ i , (i - 1) % num_qubits ])
            if graph == 'chain_reverse':
                graph_edges.append( [ (i + 1) % num_qubits , i ] )

    else:
        graph_edges = special_graphs[num_qubits][graph]

    graph_weights = np.zeros( len(graph_edges) )
    graph_edges = np.stack( graph_edges )

    for i in range(num_qubits):
        mask = graph_edges[:, -1] == i
        recp = np.sum(mask)
        if recp > 0:
            graph_weights[mask] = np.pi/( recp*3 )

    return {
        'edges'     :   graph_edges,
        'weights'   :   graph_weights
    }

