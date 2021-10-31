from dataclasses import dataclass

from tectosaur2.laplace2d import LaplaceKernel
from tectosaur2.mesh import PanelSurface

@dataclass()
class Integral:
    K: LaplaceKernel
    src: PanelSurface
    on_src_direction: float = 1.0
    d_cutoff: float = None
    d_up: float = None
    d_qbx: float = None
    d_refine: float = None
    max_p: int = None
    

def integrate(obs_pts, *integrals, tol=1e-13, return_reports=False):
    # step 1: figure out which observation points need to use QBX
    all_src_surfs = [p[0] for p in surf_K_pairs]
    src_tree = scipy.spatial.KDTree(src.pts)
    closest_dist, closest_idx = src_tree.query(obs_pts)
    closest_panel_length = src.panel_length[closest_idx // src.panel_order]
    use_qbx = closest_dist < d_qbx * closest_panel_length

    n_qbx = np.sum(use_qbx)
    if n_qbx > 0:
        qbx_closest_pts = src.pts[closest_idx][use_qbx]
        qbx_normals = src.normals[closest_idx][use_qbx]
        qbx_obs_pts = obs_pts[use_qbx]
        qbx_L = closest_panel_length[use_qbx]

        # step 3: find expansion centers
        # TODO: account for singularities


        exp_rs = qbx_L * 0.5

        direction_dot = (
            np.sum(qbx_normals * (qbx_obs_pts - qbx_closest_pts), axis=1) / exp_rs
        )
        direction = np.sign(direction_dot)
        direction[np.abs(direction) < 1e-13] = on_src_direction

        for j in range(30):
            exp_centers = (
                qbx_closest_pts + direction[:, None] * qbx_normals * exp_rs[:, None]
            )
            dist_to_nearest_panel = src_tree.query(exp_centers)[0]
            # TODO: WRITE A TEST THAT HAS VIOLATIONS
            # The fudge factor helps avoid numerical precision issues. For example,
            # when we offset an expansion center 1.0 away from a surface node,
            # without the fudge factor this test will be checking 1.0 < 1.0, but
            # that is fragile in the face of small 1e-15 sized numerical errors.
            # By simply multiplying by 1.0001, we avoid this issue without
            # introducing any other problems.
            fudge_factor = 1.0001
            which_violations = dist_to_nearest_panel * fudge_factor < np.abs(exp_rs)
            if not which_violations.any():
                break
            exp_rs[which_violations] *= 0.75

def direct(kernel, obs_pts, src):
    return np.transpose(kernel._direct(obs_pts,src), (0,2,1))

def integrate_term(
    kernel,
    obs_pts,
    term,
    tol=1e-13,
    return_report=False
):
    if d_cutoff is None:
        d_cutoff = self.d_cutoff
    if d_up is None:
        d_up = self.d_up
    if d_qbx is None:
        d_qbx = self.d_qbx
    if d_refine is None:
        d_refine = self.d_refine

    # step 1: construct the farfield matrix!
    mat = self._direct(obs_pts, src)

    # step 2: identify QBX observation points.
    src_tree = scipy.spatial.KDTree(src.pts)
    closest_dist, closest_idx = src_tree.query(obs_pts)
    closest_panel_length = src.panel_length[closest_idx // src.panel_order]
    use_qbx = closest_dist < d_qbx * closest_panel_length

    n_qbx = np.sum(use_qbx)
    if n_qbx > 0:
        qbx_closest_pts = src.pts[closest_idx][use_qbx]
        qbx_normals = src.normals[closest_idx][use_qbx]
        qbx_obs_pts = obs_pts[use_qbx]
        qbx_L = closest_panel_length[use_qbx]

        # step 3: find expansion centers
        # TODO: account for singularities


        exp_rs = qbx_L * 0.5

        direction_dot = (
            np.sum(qbx_normals * (qbx_obs_pts - qbx_closest_pts), axis=1) / exp_rs
        )
        direction = np.sign(direction_dot)
        direction[np.abs(direction) < 1e-13] = on_src_direction

        for j in range(30):
            exp_centers = (
                qbx_closest_pts + direction[:, None] * qbx_normals * exp_rs[:, None]
            )
            dist_to_nearest_panel = src_tree.query(exp_centers)[0]
            # TODO: WRITE A TEST THAT HAS VIOLATIONS
            # The fudge factor helps avoid numerical precision issues. For example,
            # when we offset an expansion center 1.0 away from a surface node,
            # without the fudge factor this test will be checking 1.0 < 1.0, but
            # that is fragile in the face of small 1e-15 sized numerical errors.
            # By simply multiplying by 1.0001, we avoid this issue without
            # introducing any other problems.
            fudge_factor = 1.0001
            which_violations = dist_to_nearest_panel * fudge_factor < np.abs(exp_rs)
            if not which_violations.any():
                break
            exp_rs[which_violations] *= 0.75

        # step 4: find which source panels need to use QBX
        (
            qbx_panels,
            qbx_panel_starts,
            qbx_panel_obs_pts,
            qbx_panel_obs_pt_starts,
        ) = identify_nearfield_panels(
            exp_centers, d_cutoff * qbx_L, src_tree, src.panel_order
        )

        # step 5: QBX integrals
        # TODO: This could be replaced by a sparse local matrix.
        qbx_mat = np.zeros((qbx_obs_pts.shape[0], src.n_pts, self.ndim))
        p, kappa_too_small = local_qbx_integrals(
            self.exp_deriv,
            self.eval_deriv,
            qbx_mat,
            qbx_obs_pts,
            src,
            exp_centers,
            exp_rs,
            max_p,
            tol,
            d_refine,
            qbx_panels,
            qbx_panel_starts,
        )
        if np.any(kappa_too_small):
            warnings.warn("Some integrals diverged because kappa is too small.")

        # step 6: subtract off the direct term whenever a QBX integral is used.
        self._nearfield(
            qbx_mat,
            qbx_obs_pts,
            src,
            qbx_panel_obs_pts,
            qbx_panel_obs_pt_starts,
            -1.0,
            0.0,
        )
        mat[use_qbx] += qbx_mat

    # step 7: nearfield integrals
    use_nearfield = (closest_dist < d_up * closest_panel_length) & (~use_qbx)
    n_nearfield = np.sum(use_nearfield)

    if n_nearfield > 0:
        nearfield_obs_pts = obs_pts[use_nearfield]
        nearfield_L = closest_panel_length[use_nearfield]

        obs_tree = scipy.spatial.KDTree(nearfield_obs_pts)
        panel_obs_pts = obs_tree.query_ball_point(
            src.panel_centers, d_up * src.panel_length
        )
        panel_obs_pts_starts = np.zeros(src.n_panels + 1, dtype=int)
        panel_obs_pts_starts[1:] = np.cumsum([len(p) for p in panel_obs_pts])
        panel_obs_pts = np.concatenate(panel_obs_pts, dtype=int, casting="unsafe")

        nearfield_mat = np.zeros((nearfield_obs_pts.shape[0], src.n_pts, self.ndim))
        self._nearfield(
            nearfield_mat,
            nearfield_obs_pts,
            src,
            panel_obs_pts,
            panel_obs_pts_starts,
            1.0,
            d_refine,
        )

        # setting d_refine=0.0 prevents refinement which is what we want to
        # cancel out the direct component terms
        self._nearfield(
            nearfield_mat,
            nearfield_obs_pts,
            src,
            panel_obs_pts,
            panel_obs_pts_starts,
            -1.0,
            0.0,
        )
        mat[use_nearfield] += nearfield_mat

    if return_report:
        report = dict()
        report["n_qbx"] = n_qbx
        report["n_nearfield"] = n_nearfield
        if n_nearfield > 0:
            report["n_nearfield_panels"] = panel_obs_pts_starts[-1]
        if n_qbx > 0:
            report["n_qbx_panels"] = qbx_panel_obs_pt_starts[-1]
        for k in [
            "exp_centers",
            "exp_rs",
            "p",
            "kappa_too_small",
            "use_qbx",
        ]:
            report[k] = locals().get(k, None)
        return np.transpose(mat, (0, 2, 1)), report
    else:
        return np.transpose(mat, (0, 2, 1))