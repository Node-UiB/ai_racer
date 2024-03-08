import torch as T


class LinAlg:

    @staticmethod
    def get_rotation_matrix(angle: T.Tensor) -> T.Tensor:
        cos_angle, sin_angle = T.cos(angle), T.sin(angle)

        return T.as_tensor(
            [[cos_angle, -sin_angle], [sin_angle, cos_angle]],
            dtype=angle.dtype,
            device=angle.device,
        )

    @staticmethod
    def safe_inverse(tensor: T.Tensor) -> T.Tensor:
        determinant = (
            tensor[..., 0, 0] * tensor[..., 1, 1]
            - tensor[..., 0, 1] * tensor[..., 1, 0]
        )[..., None, None]

        inverse = T.zeros_like(tensor)

        inverse[..., 0, 0] = tensor[..., 1, 1]
        inverse[..., 0, 1] = -tensor[..., 0, 1]
        inverse[..., 1, 0] = -tensor[..., 1, 0]
        inverse[..., 1, 1] = tensor[..., 0, 0]

        return inverse / determinant

    @staticmethod
    def get_lines(positions: T.Tensor, closed: bool) -> T.Tensor:
        if closed:
            a = positions
            ab = T.roll(positions, -1, 0) - positions

        else:
            a = positions[:-1]
            ab = positions[1:] - positions[:-1]

        return T.concat((a[..., None], ab[..., None]), dim=-1)

    @staticmethod
    def get_lines_mesh(lines_0: T.Tensor, lines_1: T.Tensor):
        lines_0 = lines_0[:, None].repeat(1, lines_1.size(0), 1, 1)
        lines_1 = lines_1[None].repeat(lines_0.size(0), 1, 1, 1)

        return lines_0, lines_1

    @staticmethod
    def get_line_matrices(lines_0: T.Tensor, lines_1: T.Tensor) -> T.Tensor:
        M = T.concat((lines_0[..., 1, None], lines_1[..., 1, None]), dim=-1)

        return M

    @staticmethod
    def get_intersecting_mask(ts: T.Tensor) -> T.Tensor:
        big_mask = T.all(ts >= 0.0, dim=-1)
        small_mask = T.all(ts < 1.0, dim=-1)

        intersecting_mask = T.all(
            T.concat((big_mask[..., None], small_mask[..., None]), dim=-1), dim=-1
        )

        return intersecting_mask

    @staticmethod
    def get_truncated_depth(lines_0: T.Tensor, lines_1: T.Tensor) -> T.Tensor:
        lines_0, lines_1 = LinAlg.get_lines_mesh(lines_0, lines_1)

        M = LinAlg.get_line_matrices(lines_0, lines_1)
        M_inv = LinAlg.safe_inverse(M)

        ts = (M_inv @ (lines_1[..., 0] + lines_1[..., 1] - lines_0[..., 0])[..., None])[
            ..., 0
        ]

        intersecting_mask = LinAlg.get_intersecting_mask(ts)
        intersecting_any_mask = T.any(intersecting_mask, dim=-1)

        ts[~intersecting_mask] = T.finfo(lines_0.dtype).max

        depths = T.ones((lines_0.size(0),), dtype=lines_0.dtype, device=lines_0.device)
        depths[intersecting_any_mask] = T.min(
            ts[intersecting_any_mask][..., 0], dim=-1
        )[0]

        return depths

    @staticmethod
    def get_intersection_points(lines_0: T.Tensor, lines_1: T.Tensor) -> T.Tensor:
        lines_0, lines_1 = LinAlg.get_lines_mesh(lines_0, lines_1)

        M = LinAlg.get_line_matrices(lines_0, lines_1)
        M_inv = LinAlg.safe_inverse(M)

        ts = (M_inv @ (lines_1[..., 0] + lines_1[..., 1] - lines_0[..., 0])[..., None])[
            ..., 0
        ]

        intersecting_mask = LinAlg.get_intersecting_mask(ts)

        ts[..., 1] = 0.0
        intersection_points = (M[intersecting_mask] @ ts[intersecting_mask][..., None])[
            ..., 0
        ] + lines_0[intersecting_mask][..., 0]

        return intersection_points

    @staticmethod
    def intersecting(lines_0: T.Tensor, lines_1: T.Tensor) -> T.Tensor:
        lines_0, lines_1 = LinAlg.get_lines_mesh(lines_0, lines_1)

        M = LinAlg.get_line_matrices(lines_0, lines_1)
        M_inv = LinAlg.safe_inverse(M)

        ts = (M_inv @ (lines_1[..., 0] + lines_1[..., 1] - lines_0[..., 0])[..., None])[
            ..., 0
        ]

        intersecting_mask = LinAlg.get_intersecting_mask(ts)

        return T.any(intersecting_mask)
    
    @staticmethod
    def get_distance_to_lines(position : T.Tensor, lines : T.Tensor) -> T.Tensor:
        a = lines[..., 0]
        ab = lines[..., 1]

        ab_square = T.sum(ab ** 2, dim=-1)

        ts = T.sum((position[None] - a) * ab, dim=-1) / ab_square

        ts_too_small_mask = ts < 0
        ts_too_big_mask = ts > 1

        ts[ts_too_small_mask] = 0.0
        ts[ts_too_big_mask] = 1.0

        ds = T.sqrt(T.sum((position[None] - a - ts[..., None] * ab) ** 2, dim=-1))

        return ds
    
    @staticmethod
    def get_distance_along_lines(position : T.Tensor, lines : T.Tensor) -> T.Tensor:
        a = lines[..., 0]
        ab = lines[..., 1]

        ab_square = T.sum(ab ** 2, dim=-1)

        ts = T.sum((position[None] - a) * ab, dim=-1) / ab_square

        ts_too_small_mask = ts < 0
        ts_too_big_mask = ts > 1

        ts[ts_too_small_mask] = 0.0
        ts[ts_too_big_mask] = 1.0

        distance_to_line = T.sqrt(T.sum((position[None] - a - ts[..., None] * ab) ** 2, dim=-1))
        nearest_line_index = T.argmin(distance_to_line, dim=0)
        distance_along_line = ts[nearest_line_index] * T.sqrt(ab_square[nearest_line_index])

        return nearest_line_index, distance_along_line
    