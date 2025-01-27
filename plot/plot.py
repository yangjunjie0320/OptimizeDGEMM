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

    for i, k in enumerate(label):
        dd[k] = xx[:, i+1]

    for k, v in dd.items():
        ax.plot(xx[:, 0], v, label=k, marker='o')

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_xlabel('L')
    ax.set_ylabel('GFLOPS')
    ax.set_xlim(xx[:, 0].min(), xx[:, 0].max())
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')

if __name__ == "__main__":
    plot(sys.argv[1], sys.argv[2])
