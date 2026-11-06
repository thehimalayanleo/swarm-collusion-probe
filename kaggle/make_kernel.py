"""Build a private Kaggle GPU kernel for one SWARM-1 plan line: the branch's collusion_probe.py is
embedded (sha256-checked at run time) ahead of that line's driver.
Usage: python kaggle/make_kernel.py <line> <kaggle-user>, then kaggle kernels push -p kaggle/build/<line>"""
import base64, hashlib, json, os, sys

DRIVERS = {
    "s2": '''
# S2: negative control at full size. Random-init Qwen2.5-1.5B, real tokenizer, layer 14, 40 functions.
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
name = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
torch.manual_seed(0)
model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(name)).to(device="cuda", dtype=torch.float16).eval()
model.config._name_or_path = "random-init " + name
res, line = cp.run(model, tok, out="results/s2-random-init", n_funcs=40, layer=14, batch=32)
print("S2 pass:", max(res["k"][3]["omp16"][0], res["k"][3]["dense"][0]) <= 0.65)
''',
}

HEAD = '''import base64, hashlib, sys, time
code = base64.b64decode("{b64}").decode()
assert hashlib.sha256(code.encode()).hexdigest() == "{sha}", "embedded code differs from the branch"
open("collusion_probe.py", "w").write(code)
sys.path.insert(0, ".")
import torch, transformers, sklearn
import collusion_probe as cp
print("code sha256 {sha}; torch", torch.__version__, "transformers", transformers.__version__,
      "sklearn", sklearn.__version__, "gpu", torch.cuda.get_device_name(0))
t0 = time.time()
'''

if __name__ == "__main__":
    line, user = sys.argv[1], sys.argv[2]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = open(os.path.join(root, "collusion_probe.py")).read()
    sha = hashlib.sha256(code.encode()).hexdigest()
    out = os.path.join(root, "kaggle", "build", line)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "run.py"), "w") as fh:
        fh.write(HEAD.format(b64=base64.b64encode(code.encode()).decode(), sha=sha) + DRIVERS[line]
                 + '\nprint(f"minutes {(time.time() - t0) / 60:.1f}")\n')
    with open(os.path.join(out, "kernel-metadata.json"), "w") as fh:
        json.dump({"id": f"{user}/swarm1-{line}", "title": f"swarm1-{line}", "code_file": "run.py",
                   "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
                   "enable_internet": True, "dataset_sources": [], "competition_sources": [],
                   "kernel_sources": []}, fh, indent=1)
    print("built", out, "code sha256", sha[:16])
