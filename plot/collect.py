import re, os, sys

# Read in the log file
log_file = sys.argv[1]
out_file = sys.argv[2]

assert os.path.exists(log_file), f"Log file {log_file} does not exist"

with open(log_file, 'r') as f:
    log = f.read()

# Parse the log to extract the data
data = {}
current_l = None
current_impl = None

for line in log.split('\n'):
    if line.startswith('Running'):
        m = re.match(r'Running (\S+) with arguments (\d+) ...', line)
        current_impl = m.group(1).replace('main-dgemm-', '')
        current_l = int(m.group(2))
        if current_l not in data:
            data[current_l] = {}
    elif line.startswith('MM_REF'):  
        # extract gflops and std_err use regex
        # example: `MM_REF t = 6.75e-06 sec, GFLOPS =   9.72 +/-   0.31`
        m = line.split()
        data[current_l]['BLAS'] = float(m[7])
        data[current_l]['BLAS_stderr'] = float(m[9])
    elif line.startswith('MM_SOL'):
        if current_impl.endswith('.x'):
            current_impl = current_impl[:-2]

        m = line.split()
        data[current_l][current_impl] = float(m[7])
        data[current_l][current_impl + '_stderr'] = float(m[9])

# Print the results, add the std_err to the data
impls = ['BLAS'] + sorted(set([impl for l in data.values() for impl in l if impl != 'BLAS']))

# format the output
lmax = max([len(impl) for impl in impls])
title = ("# %6s, " % "L" + f"%{lmax}s, " * (len(impls))) % tuple(impls)

# print the title
with open(out_file, 'w') as f:
    f.write(title[:-2] + '\n')
    for l in sorted(data.keys()):
        info = "%8d, " % l
        for impl in impls:
            x = (f"%{lmax}.2f, ")
            info += x % data[l].get(impl, 'nan')
        info = info[:-2]
        f.write(info + '\n')

with open(out_file, 'r') as f:
    print(f.read())