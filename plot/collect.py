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
        current_impl = m.group(1).replace('main-sgemm-', '')
        current_l = int(m.group(2))
        if current_l not in data:
            data[current_l] = {}
    elif line.startswith('MM_REF'):  
        data[current_l]['BLAS'] = float(line.split('GFLOPS = ')[1])
    elif line.startswith('MM_SOL'):
        data[current_l][current_impl] = float(line.split('GFLOPS = ')[1])

# Print the results
impls = ['BLAS'] + sorted(set([impl for l in data.values() for impl in l if impl != 'BLAS']))

# format the output
# print('L,' + ','.join(impls))
title = ("# %6s, " % "L" + "%12s, " * (len(impls))) % tuple(impls)
print(title[:-2])
for l in sorted(data.keys()):
    print(f"{l:8d}, " + ', '.join([f"{data[l].get(impl, 'nan'): 12.2f}" for impl in impls]))

with open(out_file, 'w') as f:
    f.write(title[:-2] + '\n')
    for l in sorted(data.keys()):
        f.write(f"{l:8d}, " + ', '.join([f"{data[l].get(impl, 'nan'): 12.2f}" for impl in impls]) + '\n')
