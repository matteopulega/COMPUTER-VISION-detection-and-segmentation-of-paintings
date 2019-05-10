import numpy as np


def perp(a):
    p = np.empty_like(a)
    p[:, 0] = -a[:, 1]
    p[:, 1] = a[:, 0]
    return p


def lines_intersection(a, b):
    da = a[:, 0] - a[:, 1]
    db = b[:, 0] - b[:, 1]
    dp = a[:, 0] - b[:, 0]
    dap = perp(da)
    denom = np.dot(dap, db.T)
    num = np.dot(dap, dp.T)
    val = (num / denom.astype(float))
    val = val @ db
    val = val + b[:, 0]
    return val


def to_cartesian_equation(lines):
    if len(lines.shape) != 3:
        lines = lines.reshape((1, -1, 2))
    diff = np.diff(lines, axis=1).reshape((-1, 2))
    idxs_0 = diff[:, 0] == 0
    equations = np.empty(shape=(lines.shape[0], 2), dtype=np.float)
    equations[idxs_0, 0] = np.inf
    equations[idxs_0, 1] = lines[idxs_0, 0, 0]
    idxs_not_0 = diff[:, 0] != 0
    diff = diff[idxs_not_0]
    m = np.true_divide(diff[:, 1], diff[:, 0])
    q = lines[idxs_not_0, 0, 1] - m * lines[idxs_not_0, 0, 0]
    equations[idxs_not_0, 0] = m
    equations[idxs_not_0, 1] = q
    return equations


def compute_in_point(equations, x):
    points = np.empty_like(equations, dtype=np.float)
    infs = np.isinf(equations[:, 0])
    if np.sum(infs) > 0:
        verticals = equations[infs]
        verticals = verticals[:, 1].reshape((-1, 1))
        xs = np.repeat(x, len(verticals)).reshape((-1, 1))
        points[infs] = np.hstack((verticals, xs))
    not_infs = np.invert(infs)
    if np.sum(not_infs) > 0:
        not_verticals = equations[not_infs]
        xs = np.repeat(x, len(not_verticals)).reshape((-1, 1))
        points[not_infs] = np.hstack((xs, (not_verticals[:, 0] * x + not_verticals[:, 1]).reshape((-1, 1))))
    return points
