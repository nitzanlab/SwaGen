import numpy as np

def arrange_agents_in_rectangle(N, alpha=5, rotation_angle=135):
    points = []

    num_cols = int(np.ceil(np.sqrt(N)))
    num_rows = int(np.ceil(N / num_cols))

    index = 0
    for row in range(num_rows):
        for col in range(num_cols):
            if index >= N:
                break
            x = col * alpha
            y = row * alpha
            points.append((x, y))
            index += 1
        if index >= N:
            break

    points = np.array(points, dtype=np.float64)

    # Rotate points by the specified angle
    if rotation_angle != 0:
        theta = np.radians(rotation_angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        center = np.mean(points, axis=0)
        points -= center
        points = points @ R.T
        points += center

    # Ensure all points are non-negative
    min_x, min_y = points.min(axis=0)
    points[:, 0] -= min_x if min_x < 0 else 0
    points[:, 1] -= min_y if min_y < 0 else 0

    return points


def arrange_agents_in_arrow(N, alpha=5):
    points = []

    num_rows = int(np.ceil((np.sqrt(8 * N + 1) - 1) / 2))

    index = 0
    for row in range(num_rows):
        for col in range(row + 1):
            if index >= N:
                break
            x = col * alpha
            y = row * alpha
            points.append((x, y))
            index += 1
        if index >= N:
            break

    points = np.array(points, dtype=np.float64)

    # Rotate points by 270 degrees
    theta = np.radians(270)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    points = points @ R.T

    # Ensure all points are non-negative
    min_x, min_y = points.min(axis=0)
    points[:, 0] -= min_x if min_x < 0 else 0
    points[:, 1] -= min_y if min_y < 0 else 0

    return points

def arrange_agents_in_v(N, alpha=5):
    points = []

    half_points = N // 2

    for i in range(half_points):
        x = i * alpha
        y = i * alpha
        points.append((x, y))

    for i in range(half_points):
        x = i * alpha
        y = -i * alpha
        points.append((x, y))

    if N % 2 == 1:
        points.append((0, 0))

    points = np.array(points, dtype=np.float64)

    # Rotate points by 180 degrees
    theta = np.radians(225)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    points = points @ R.T

    # Ensure all points are non-negative
    min_x, min_y = points.min(axis=0)
    points[:, 0] -= min_x if min_x < 0 else 0
    points[:, 1] -= min_y if min_y < 0 else 0

    return points


import numpy as np


def arrange_agents_in_kite(N, alpha=5, rotation_angle=0, curve_factor=0.25, block_curve_gap=0.5):
    """
    Arrange N agents in a kite-like shape:
    a block of points on top and a curved line of points below it, connected to the block.

    Args:
        N (int): Number of agents.
        alpha (float): Distance between adjacent points.
        rotation_angle (float): Angle to rotate the entire shape (in degrees).
        curve_factor (float): Factor controlling the curvature of the line below the block.
        block_curve_gap (float): Factor controlling the vertical distance between the block and the curve.

    Returns:
        np.ndarray: Array of shape (N, 2) representing the coordinates of the points.
    """
    points = []

    # Divide points between the block and the curved line
    num_block_points = max(1, N // 2)  # At least 1 point in the block
    num_line_points = N - num_block_points  # Remaining points in the curved line

    # Create the block of points (placed at the top)
    num_block_rows = int(np.ceil(np.sqrt(num_block_points)))
    num_block_cols = int(np.ceil(num_block_points / num_block_rows))

    index = 0
    for row in range(num_block_rows):
        for col in range(num_block_cols):
            if index >= num_block_points:
                break
            x = col * alpha - (num_block_cols * alpha) / 2  # Center the block horizontally
            y = row * alpha + block_curve_gap * alpha  # Block placed above the curve
            points.append((x, y))
            index += 1
        if index >= num_block_points:
            break

    # Create the curved line below the block
    for i in range(num_line_points):
        x = -i * alpha * np.cos(curve_factor * i)  # Introduce curvature
        y = -i * alpha  # Downward motion for the curve
        points.append((x, y))

    points = np.array(points, dtype=np.float64)

    # Rotate points by the specified angle
    if rotation_angle != 0:
        theta = np.radians(rotation_angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        center = np.mean(points, axis=0)
        points -= center
        points = points @ R.T
        points += center

    # Ensure all points are non-negative
    min_x, min_y = points.min(axis=0)
    points[:, 0] -= min_x if min_x < 0 else 0
    points[:, 1] -= min_y if min_y < 0 else 0

    return points




def arrange_agents_in_v_triangle_wave(N, alpha=5, wavelength=23):
    points = []

    half_points = N // 2

    # Sharper triangles: Increase the amplitude of the triangle wave relative to wavelength
    amplitude = wavelength 

    # Generate the upper arm of the V-shape as a sharp triangle wave (upper-right direction)
    for i in range(half_points):
        x = i * alpha
        y = (i * alpha) % wavelength
        if (i * alpha // wavelength) % 2 == 1:
            y = wavelength - y
        y *= amplitude / wavelength  # Scale y for sharper peaks
        points.append((x, y))

    # Generate the lower arm of the V-shape as a sharp triangle wave (lower-left direction)
    for i in range(half_points):
        y = i * alpha
        x = (i * alpha) % wavelength
        if (i * alpha // wavelength) % 2 == 1:
            x = wavelength - x
        x *= amplitude / wavelength  # Scale x for sharper peaks
        points.append((-x, -y))  # Ensure orthogonality by mirroring direction

    if N % 2 == 1:
        points.append((0, 0))

    points = np.array(points, dtype=np.float64)

    # Rotate points to face the upper-right direction
    theta = np.radians(-90)  # Adjust rotation for the upper-right orientation
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    points = points @ R.T

    # Ensure all points are non-negative
    min_x, min_y = points.min(axis=0)
    points[:, 0] -= min_x if min_x < 0 else 0
    points[:, 1] -= min_y if min_y < 0 else 0

    return points
def arrange_agents_in_kite2(N, alpha=5, rotation_angle=0, curve_factor=0.025, block_curve_gap=0.5):
    """
    Arrange N agents in a kite-like shape:
    a block of points on top and a curved line of points below it, connected to the block.

    Args:
        N (int): Number of agents.
        alpha (float): Distance between adjacent points.
        rotation_angle (float): Angle to rotate the entire shape (in degrees).
        curve_factor (float): Factor controlling the curvature of the line below the block.
        block_curve_gap (float): Factor controlling the vertical distance between the block and the curve.

    Returns:
        np.ndarray: Array of shape (N, 2) representing the coordinates of the points.
    """
    points = []

    # Divide points between the block and the curved line
    num_block_points = max(1, N*2 // 3)  # At least 1 point in the block
    num_line_points = N - num_block_points  # Remaining points in the curved line

    # Create the block of points (placed at the top)
    num_block_rows = int(np.ceil(np.sqrt(num_block_points)))
    num_block_cols = int(np.ceil(num_block_points / num_block_rows))

    index = 0
    for row in range(num_block_rows):
        for col in range(num_block_cols):
            if index >= num_block_points:
                break
            x = col * alpha - (num_block_cols * alpha) / 2  # Center the block horizontally
            y = row * alpha + block_curve_gap * alpha  # Block placed above the curve
            points.append((x, y))
            index += 1
        if index >= num_block_points:
            break

    # Create the curved line below the block
    for i in range(num_line_points):
        x = -i * alpha * np.cos(curve_factor * i)  # Introduce curvature
        y = -i * alpha  # Downward motion for the curve
        points.append((x, y))

    points = np.array(points, dtype=np.float64)

    # Rotate points by the specified angle
    if rotation_angle != 0:
        theta = np.radians(rotation_angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        center = np.mean(points, axis=0)
        points -= center
        points = points @ R.T
        points += center

    # Ensure all points are non-negative
    min_x, min_y = points.min(axis=0)
    points[:, 0] -= min_x if min_x < 0 else 0
    points[:, 1] -= min_y if min_y < 0 else 0

    return points

def arrange_agents_in_kite3(N, alpha=5, rotation_angle=0, curve_factor=0.01, block_curve_gap=0.5):
    """
    Arrange N agents in a kite-like shape:
    a block of points on top and a curved line of points below it, connected to the block.

    Args:
        N (int): Number of agents.
        alpha (float): Distance between adjacent points.
        rotation_angle (float): Angle to rotate the entire shape (in degrees).
        curve_factor (float): Factor controlling the curvature of the line below the block.
        block_curve_gap (float): Factor controlling the vertical distance between the block and the curve.

    Returns:
        np.ndarray: Array of shape (N, 2) representing the coordinates of the points.
    """
    points = []

    # Divide points between the block and the curved line
    num_block_points = max(1, N*3 // 4)  # At least 1 point in the block
    num_line_points = N - num_block_points  # Remaining points in the curved line

    # Create the block of points (placed at the top)
    num_block_rows = int(np.ceil(np.sqrt(num_block_points)))
    num_block_cols = int(np.ceil(num_block_points / num_block_rows))

    index = 0
    for row in range(num_block_rows):
        for col in range(num_block_cols):
            if index >= num_block_points:
                break
            x = col * alpha - (num_block_cols * alpha) / 2  # Center the block horizontally
            y = row * alpha + block_curve_gap * alpha  # Block placed above the curve
            points.append((x, y))
            index += 1
        if index >= num_block_points:
            break

    # Create the curved line below the block
    for i in range(num_line_points):
        x = -i * alpha * np.cos(curve_factor * i)  # Introduce curvature
        y = -i * alpha  # Downward motion for the curve
        points.append((x, y))

    points = np.array(points, dtype=np.float64)

    # Rotate points by the specified angle
    if rotation_angle != 0:
        theta = np.radians(rotation_angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        center = np.mean(points, axis=0)
        points -= center
        points = points @ R.T
        points += center

    # Ensure all points are non-negative
    min_x, min_y = points.min(axis=0)
    points[:, 0] -= min_x if min_x < 0 else 0
    points[:, 1] -= min_y if min_y < 0 else 0

    return points



def arrange_agents_random(N, alpha, max_attempts=1000):
    """
    Randomly arrange N agents so that:
      - Each new agent (except the first) is placed exactly α away from a randomly chosen already placed agent.
      - No two agents are closer than α.
      
    This ensures that every agent (after the first) has a neighbor exactly at distance α,
    and every pair of agents maintains a separation of at least α.

    Parameters:
        N (int): Total number of agents.
        alpha (float): The distance used for placement and the minimum allowed distance between agents.
        max_attempts (int): Maximum number of attempts to place an agent before giving up.
    
    Returns:
        np.ndarray: An array of shape (N, 2) with the (x, y) coordinates of the agents.
        
    Raises:
        RuntimeError: If a new agent cannot be placed after max_attempts.
    """
    points = []
    # Place the first agent arbitrarily at the origin.
    points.append(np.array([0.0, 0.0]))
    
    # For every subsequent agent, choose an anchor from the already placed agents.
    for i in range(1, N):
        placed = False
        attempts = 0
        while not placed and attempts < max_attempts:
            attempts += 1
            # Randomly pick an anchor from the existing agents.
            anchor = points[np.random.randint(0, len(points))]
            # Sample a random angle uniformly from 0 to 2π.
            theta = np.random.uniform(0, 2*np.pi)
            # Candidate position exactly α away from the chosen anchor.
            candidate = anchor + np.array([alpha * np.cos(theta), alpha * np.sin(theta)])
            # Check that candidate is at least α away from every already placed agent.
            distances = np.linalg.norm(np.array(points) - candidate, axis=1)
            # Allow equality with α (since the anchor is exactly at α)
            if np.all(distances >= alpha - 1e-8):
                points.append(candidate)
                placed = True
        if not placed:
            raise RuntimeError(f"Could not place agent {i} after {max_attempts} attempts. "
                               "Consider adjusting α or increasing max_attempts.")
    
    points = np.array(points)
    
    # (Optional) Translate points so that all coordinates are non-negative.
    min_coords = points.min(axis=0)
    if np.any(min_coords < 0):
        points -= min_coords
    
    return points


