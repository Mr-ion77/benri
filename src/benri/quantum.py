import torch

class FalseQubitSystem(torch.tensor):
    """
    A simple False Qubit class extending torch.tensor to represent a "qubit state" from its pseudo-coordinates on the Bloch sphere:
    Attributes:
        state (torch.tensor): A tensor of shape (2,) representing the (height, depth) coordinates on the Bloch sphere.
        height (float): The latitude (-1 to 1) -> defaults to -1 (|0> state).
        depth (float): The longitude (-1 to 1) -> defaults to -1 (Greenwich).
    """
    def __init__(self, states=torch.tensor([-1.0, -1.0])): # Default state |0> represented as (-1,0)
        assert states.shape[-1] == 2, "State must be a tensor of shape (..., 2) representing (height, depth) coordinates."
        super().__init__(states)

    def get_coordinates(self):
        """
        Returns the (height, depth) coordinates of the qubit state of given qubit.
        Returns:
            tuple: A tuple containing (height, depth).
        """
        height, depth = self[...,0].item(), self[...,1].item()
        return height, depth
    
    def measure(self):
        """
        Simulates a measurement of the qubit state.
        Returns:
            int: 0 or 1 based on the qubit state probabilities.
        """
        coordinates = self.get_coordinates()
        normalized = torch.nn.functional.normalize(coordinates, p = 2, dim = -1)

        prob_0 = (1 + normalized[..., 0]) / 2  # Probability of measuring |0>
        sampled = torch.rand_like(prob_0)
        return sampled >= prob_0
    
def Rotation(system, qubit, axis, angle):
    """
    Applies a rotation to the given qubit around the specified axis by the given angle.
    Args:
        system
        qubit (int): The qubit to be rotated.
        axis (str): The axis of rotation ('X', 'Y', or 'Z').
        angle (float): The angle of rotation in radians.
    Returns:
        FalseQubit: The rotated qubit.
    """
    vheight, vdepth = system.get_coordinates()
    height, depth = vheight[...,qubit], vdepth[...,qubit]
    
    if axis == 'X':
        new_height =   height * torch.sin(angle)
        new_depth   =   depth   * torch.cos(angle)
    elif axis == 'Y':
        new_height =   height * torch.cos(angle)
        new_depth   =   depth
    elif axis == 'Z':
        new_height =   height
        new_depth   =   depth   * torch.cos(angle) 
    else:
        raise ValueError("Axis must be 'X', 'Y', or 'Z'.")
    
    vheight[...,qubit], vdepth[...,qubit] = new_height, new_depth
    
    return FalseQubitSystem(torch.tensor([new_height, new_depth]))

    