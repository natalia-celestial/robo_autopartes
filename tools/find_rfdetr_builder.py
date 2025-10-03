# tools/find_rfdetr_builder.py
import importlib, inspect, os, pkgutil, sys
import torch.nn as nn

import rfdetr
pkg_dir = os.path.dirname(rfdetr.__file__)
print("rfdetr package at:", pkg_dir)

candidates = []
for _, modname, _ in pkgutil.walk_packages([pkg_dir], prefix="rfdetr."):
    try:
        m = importlib.import_module(modname)
    except Exception:
        continue
    for name, obj in vars(m).items():
        # look for a function that returns an nn.Module OR a class that is an nn.Module
        if inspect.isfunction(obj):
            try:
                sig = inspect.signature(obj)
                if any(k in sig.parameters for k in ("num_classes","classes")):
                    candidates.append((modname, name, "func"))
            except Exception:
                pass
        if inspect.isclass(obj) and issubclass(obj, nn.Module):
            candidates.append((modname, name, "class"))

print("Found candidates:")
for mod, name, kind in candidates:
    if "detr" in (mod+name).lower():
        print(f"  * {kind}: from {mod} import {name}")
