import subprocess


# call "nvidia-smi" to get memory
def report():
    l1 = len("| 30%   35C    P0    62W / 350W |      0")
    l0 = len("| 30%   35C    P0    62W / 350W |")
    out = subprocess.check_output("nvidia-smi", shell=True)
    if type(out) != type("a"): out = out.decode('utf-8')
    mem = int(out.split("\n")[9][l0:l1])
    print(f'inference memory: {mem} MB')
