import os, sys, numpy
from matplotlib import pyplot as plt

def plot(data, out):
    assert os.path.exists(data), f"Data file {data} does not exist"
    lines = open(data).readlines()[0].split(',')
    
    label = lines[1:]
    label = [l.strip() for l in label]

    fig, ax = plt.subplots(figsize=(6, 3))

    dd = {}
    xx = numpy.loadtxt(data, delimiter=',', skiprows=1)
    assert len(label) == xx.shape[1] - 1

    for i, k in enumerate(label):
        dd[k] = xx[:, i+1]

    # plot the data and the error bar
    for k, v in dd.items():
        if k.endswith('_stderr'):
            continue

        # get the color of the line
        l = ax.plot(xx[:, 0], v, label=k, marker='')
        ax.errorbar(xx[:, 0], v, yerr=dd[k + '_stderr'], fmt='none', ecolor=l[0].get_color(), capsize=2)

    #     # calculate the distribution of v
    #     m = numpy.mean(v)
    #     s = numpy.std(v)
    #     mm[k] = m
    #     ss[k] = s

    # # sort mm by value
    # mm = sorted(mm.items(), key=lambda x: x[1])

    # for k, v in mm:
    #     print(f"{k}: {v} +/- {ss[k]}")

    # # plot the distribution of v
    # fig, ax = plt.subplots(figsize=(6, 3))
    # for k, v in mm:
    #     # if not  k == 'BLAS':
    #     #     continue
    #     # ax.hist(dd[k], bins=20, label=k)
    #     # # Plot the normal distribution
    #     m = numpy.mean(dd[k])
    #     s = numpy.std(dd[k])
    #     x = numpy.linspace(0, 100, 100)
    #     y = numpy.exp(-(x-m)**2/(2*s**2))/(s*numpy.sqrt(2*numpy.pi))
        
    #     ax.plot(x, y, label=k)

    ax.set_xlabel('L')
    ax.set_xlim(xx[:, 0].min(), xx[:, 0].max())
    ax.set_ylim(0.0, 20.0)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight', pad_inches=0.0, dpi=300)

if __name__ == "__main__":
    plot(sys.argv[1], sys.argv[2])
