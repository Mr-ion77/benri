import torch
    
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

    