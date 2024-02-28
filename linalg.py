import torch as T

import matplotlib.pyplot as plt

def get_rotation_matrix(angle : T.Tensor) -> T.Tensor:
    cos_angle, sin_angle = T.cos(angle), T.sin(angle)

    return T.as_tensor([[cos_angle, -sin_angle], 
                        [sin_angle, cos_angle]], 
                       dtype=angle.dtype, 
                       device=angle.device)


def safe_inverse(tensor : T.Tensor) -> T.Tensor:
    determinant = (tensor[..., 0, 0] * tensor[..., 1, 1] - tensor[..., 0, 1] * tensor[..., 1, 0])[..., None, None]

    inverse = T.zeros_like(tensor)

    inverse[..., 0, 0] = tensor[..., 1, 1]
    inverse[..., 0, 1] = -tensor[..., 0, 1]
    inverse[..., 1, 0] = -tensor[..., 1, 0]
    inverse[..., 1, 1] = tensor[..., 0, 0]

    return inverse / determinant


def get_lines(positions : T.Tensor, closed : bool) -> T.Tensor:
    if(closed == True):
        a = positions
        ab = T.roll(positions, -1, 0) - positions

    else:
        a = positions[:-1]
        ab = positions[1:] - positions[:-1]
        
    return T.concat((a[..., None], ab[..., None]), dim=-1)


def get_lines_mesh(lines_0 : T.Tensor, lines_1 : T.Tensor):
    lines_0 = lines_0[:, None].repeat(1, lines_1.size(0), 1, 1)
    lines_1 = lines_1[None].repeat(lines_0.size(0), 1, 1, 1)

    return lines_0, lines_1


def get_line_matrices(lines_0 : T.Tensor, lines_1 : T.Tensor) -> T.Tensor:
    M = T.concat((lines_0[..., 1, None], lines_1[..., 1, None]), dim=-1)

    return M


def get_intersecting_mask(ts : T.Tensor) -> T.Tensor:
    big_mask = T.all(ts >= 0.0, dim=-1)
    small_mask = T.all(ts < 1.0, dim=-1)

    intersecting_mask = T.all(T.concat((big_mask[..., None], small_mask[..., None]), dim=-1), dim=-1)

    return intersecting_mask
    

def get_truncated_depth(lines_0 : T.Tensor, lines_1 : T.Tensor) -> T.Tensor:
    lines_0, lines_1 = get_lines_mesh(lines_0, lines_1)

    M = get_line_matrices(lines_0, lines_1)
    M_inv = safe_inverse(M)

    ts = (M_inv @ (lines_1[..., 0] + lines_1[..., 1] - lines_0[..., 0])[..., None])[..., 0]

    intersecting_mask = get_intersecting_mask(ts)
    intersecting_any_mask = T.any(intersecting_mask, dim=-1)
    
    ts[~intersecting_mask] = T.finfo(lines_0.dtype).max

    depths = T.ones((lines_0.size(0),), dtype=lines_0.dtype, device=lines_0.device)
    depths[intersecting_any_mask] = T.min(ts[intersecting_any_mask][..., 0], dim=-1)[0]

    return depths


def get_intersection_points(lines_0 : T.Tensor, lines_1 : T.Tensor) -> T.Tensor:
    lines_0, lines_1 = get_lines_mesh(lines_0, lines_1)

    M = get_line_matrices(lines_0, lines_1)
    M_inv = safe_inverse(M)

    ts = (M_inv @ (lines_1[..., 0] + lines_1[..., 1] - lines_0[..., 0])[..., None])[..., 0]

    intersecting_mask = get_intersecting_mask(ts)
    
    ts[..., 1] = 0.0
    intersection_points = (M[intersecting_mask] @ ts[intersecting_mask][..., None])[..., 0] + lines_0[intersecting_mask][..., 0]

    return intersection_points


def intersecting(lines_0 : T.Tensor, lines_1 : T.Tensor) -> bool:
    lines_0, lines_1 = get_lines_mesh(lines_0, lines_1)
    
    M = get_line_matrices(lines_0, lines_1)
    M_inv = safe_inverse(M)

    ts = (M_inv @ (lines_1[..., 0] + lines_1[..., 1] - lines_0[..., 0])[..., None])[..., 0]

    intersecting_mask = get_intersecting_mask(ts)
    
    return T.any(intersecting_mask)


if(__name__ == "__main__"):
    dtype = T.float64
    device = "cuda:0" if T.cuda.is_available() else "cpu"

    car_points = T.as_tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]], dtype=dtype, device=device)
    left_rail = T.as_tensor([[-2.0, 3.0], [0.0, 3.0], [1.0, -1.0], [-1.0, 3.0]], dtype=dtype, device=device)

    car_lines = get_lines(car_points, closed=False)
    left_rail_lines = get_lines(left_rail, closed=False)

    intersection_points = get_intersection_points(car_lines, left_rail_lines)

    car_points = car_points.cpu().numpy()
    left_rail = left_rail.cpu().numpy()

    intersection_points = intersection_points.cpu().numpy()

    plt.plot(car_points[:, 0], car_points[:, 1], c="red", label="Car")
    plt.plot(left_rail[:, 0], left_rail[:, 1], c="black", label="Road")
    plt.scatter(intersection_points[:, 0], intersection_points[:, 1], c="blue", label="Intersection Points")
    plt.legend()
    plt.show()