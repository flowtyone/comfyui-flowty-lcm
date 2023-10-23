import sys
import pkg_resources
import subprocess

def is_module_installed(module_name):
    installed_packages = [d.key for d in pkg_resources.working_set]
    return module_name in installed_packages

def install_module(module_name):
    subprocess.check_call([sys.executable, "-m", "pip", "install", module_name])

module_name = "diffusers"  
if not is_module_installed(module_name):
    print(f"### ComfyUI-LCM: {module_name} is not installed. Installing now...")
    install_module(module_name)
    print(f"### ComfyUI-LCM: {module_name} has been installed.")
else:
    print(f"### ComfyUI-LCM: {module_name} is already installed.")


from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
