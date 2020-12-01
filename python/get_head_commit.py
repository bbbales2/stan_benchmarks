import shlex
import subprocess

def get_head_commit(url, branch):
    cmd = f"git ls-remote --heads {url} {branch}"
    run = subprocess.run(
        shlex.split(cmd),
        capture_output=True,
    )

    if run.returncode != 0:
        print(f"stdout: {run.stdout}")
        print(f"stderr: {run.stderr}", flush=True)
        raise Exception(f"Exception running: {cmd}")

    out = run.stdout.decode('utf-8').strip().split()

    if len(out) == 0:
        print(f"stdout: {run.stdout}")
        print(f"stderr: {run.stderr}", flush=True)
        raise Exception(f"Expected git commit id, found nothing in: {cmd}")

    return out[0]