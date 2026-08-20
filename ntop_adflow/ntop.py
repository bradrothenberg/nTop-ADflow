"""Run an nTop notebook headless with ntopcl and collect the exported geometry.

Three things bite here, all confirmed on live runs:

1. The EXIT CODE IS NOT A SUCCESS SIGNAL. It has been observed as 0 on success on
   nTop 5.49/5.50 and as 72 on success on older setups. Gate on the expected output
   FILES appearing, never on the return code. A hung run still writes early artifacts,
   so the presence of a STEP or a JSON alone is not success either.

2. ntopcl can HANG after meshing: the mesh finishes, export never completes, the process
   stays alive with no progress and no error. Always wrap the call in a hard timeout and
   retry. Some design points hang deterministically at any timeout, which is a defect in
   the notebook's meshing tolerances rather than something a retry can fix.

3. `--trustnotebook` is REQUIRED if the notebook writes files or runs Run-Command blocks.
   Without it those blocks quietly no-op and you get no geometry and no error.

Discover a notebook's inputs with `template()` rather than guessing the schema.
"""
import glob
import json
import os
import subprocess


class NTopError(RuntimeError):
    pass


def _exe(exe=None):
    if exe:
        return exe
    env = os.environ.get("NTOPCL")
    if env:
        return env
    for c in (r"C:\Program Files\nTopology\nTopology\ntopcl.exe",
              "/opt/ntop/ntopcl"):
        if os.path.exists(c):
            return c
    raise NTopError("ntopcl not found; set the NTOPCL environment variable")


def _creds():
    u, p = os.environ.get("NTOP_USERNAME"), os.environ.get("NTOP_PASSWORD")
    if not u or not p:
        raise NTopError(
            "NTOP_USERNAME and NTOP_PASSWORD must be set in the environment. "
            "ntopcl logs in on every invocation; there is no cached token in headless "
            "use. Watch the shell escaping: a password stored with a stray backslash "
            "before '!' is rejected as a bad password.")
    return u, p


def template(notebook, outdir=".", exe=None, timeout=600):
    """Write input_template.json and output_template.json for a notebook.

    Do not guess a notebook's input schema. Generate it, then edit the values.
    """
    u, p = _creds()
    cmd = [_exe(exe), "-u", u, "-w", p, "--template", os.path.abspath(notebook)]
    subprocess.run(cmd, cwd=outdir, timeout=timeout, check=False,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    got = os.path.join(outdir, "input_template.json")
    if not os.path.exists(got):
        raise NTopError("ntopcl --template produced no input_template.json in %s" % outdir)
    with open(got) as f:
        return json.load(f)


def run(notebook, inputs, outdir, expect="*.obj", exe=None,
        timeout=1500, attempts=3, verbose=2):
    """Run a notebook once. Returns the list of files matching `expect`.

    Success is the appearance of files matching `expect`, not the return code.
    """
    u, p = _creds()
    os.makedirs(outdir, exist_ok=True)
    ipath = os.path.join(outdir, "input.json")
    with open(ipath, "w") as f:
        json.dump(inputs, f, indent=1)
    opath = os.path.join(outdir, "output.json")
    log = os.path.join(outdir, "ntopcl.log")

    cmd = [_exe(exe), "--username", u, "--password", p, "--trustnotebook",
           "-j", ipath, "-o", opath, os.path.abspath(notebook), "-v", str(verbose)]

    last = None
    for attempt in range(1, attempts + 1):
        for stale in glob.glob(os.path.join(outdir, expect)):
            os.remove(stale)
        try:
            r = subprocess.run(cmd, cwd=outdir, timeout=timeout,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               start_new_session=True)
            with open(log, "wb") as f:
                f.write(r.stdout or b"")
            last = "exit %d" % r.returncode
        except subprocess.TimeoutExpired:
            last = "timeout after %ds" % timeout
        hits = sorted(glob.glob(os.path.join(outdir, expect)))
        if hits:
            return hits
        print("  ntopcl attempt %d/%d produced no %s (%s)" % (attempt, attempts, expect, last))
    raise NTopError("ntopcl produced no files matching %r after %d attempts (%s). "
                    "Read %s: the warnings name the offending mesh block."
                    % (expect, attempts, last, log))
