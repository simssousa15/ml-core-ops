
# Step 1 — Create + activate a Python venv  
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

# Step 2 — Install build dependencies in the venv  
```bash
pip install -e . --no-build-isolation
```

This will:

- run Cython on `convlib.pyx`
- compile `convlib_backend.c`
- build `convlib.cpython-312-...so`
- place it in your venv’s site-packages
- allow fast reinstallation/rebuilding

This is the best setup for editing C or Cython code.

---

# Step 3 — (During development) Rebuild quickly when you change C code  
Whenever you edit:

- `convlib_backend.c`
- `convlib_backend.h`
- `convlib.pyx`

You run:

```bash
pip install -e . --no-build-isolation
```

---