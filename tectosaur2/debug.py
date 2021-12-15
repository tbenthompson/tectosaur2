import matplotlib.pyplot as plt


def plot_centers(report, xrange, yrange, failed=False):
    cs = report["exp_centers"]
    rs = report["exp_rs"]
    op = report["obs_pts"][report["use_qbx"]]
    if failed:
        cs = cs[report["integration_failed"]]
        rs = rs[report["integration_failed"]]
        op = op[report["integration_failed"]]
    for s in report["srcs"]:
        plt.plot(s.pts[:, 0], s.pts[:, 1], "r-o")
    plt.plot(op[:, 0], op[:, 1], "m.", markersize=10)
    plt.plot(cs[:, 0], cs[:, 1], "b.", markersize=10)
    for i in range(cs.shape[0]):
        if (xrange[0] < cs[i, 0] < xrange[1]) and (yrange[0] < cs[i, 1] < yrange[1]):
            plt.gca().add_patch(plt.Circle(cs[i], rs[i], color="k", fill=False))
    plt.axis("scaled")
    plt.xlim(xrange)
    plt.ylim(yrange)
