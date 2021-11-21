import matplotlib.pyplot as plt


def plot_centers(report, xrange, yrange):
    cs = report["exp_centers"]
    rs = report["exp_rs"]
    src = report["combined_src"]
    plt.plot(src.pts[:, 0], src.pts[:, 1], "r-o")
    plt.plot(cs[:, 0], cs[:, 1], "k.", markersize=10)
    for i in range(cs.shape[0]):
        if (xrange[0] < cs[i, 0] < xrange[1]) and (yrange[0] < cs[i, 1] < yrange[1]):
            plt.gca().add_patch(plt.Circle(cs[i], rs[i], color="k", fill=False))
    plt.axis("scaled")
    plt.xlim(xrange)
    plt.ylim(yrange)
