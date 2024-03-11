import os
import re
import subprocess
import warnings
from typing import List, Set

import torch
from packaging.version import Version, parse
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    ROCM_HOME,
    BuildExtension,
    CUDAExtension,
)


ROOT_DIR = os.path.dirname(__file__)

ext_modules = []
cmdclass = {}


def _is_hip() -> bool:
    return torch.version.hip is not None


def _is_cuda() -> bool:
    return torch.version.cuda is not None


if CUDA_HOME is not None or ROCM_HOME is not None:
    # vllm setup for csrc
    MAIN_CUDA_VERSION = "12.1"

    # Supported NVIDIA GPU architectures.
    NVIDIA_SUPPORTED_ARCHS = {"7.0", "7.5", "8.0", "8.6", "8.9", "9.0"}
    ROCM_SUPPORTED_ARCHS = {"gfx90a", "gfx908", "gfx906", "gfx1030", "gfx1100"}

    # Compiler flags.
    CXX_FLAGS = ["-g", "-O2", "-std=c++17"]
    NVCC_FLAGS = ["-O2", "-std=c++17"]

    ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
    CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
    NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]

    if _is_hip():
        if ROCM_HOME is None:
            raise RuntimeError(
                "Cannot find ROCM_HOME. ROCm must be available to build the package."
            )
        NVCC_FLAGS += ["-DUSE_ROCM"]

    def get_amdgpu_offload_arch():
        command = "/opt/rocm/llvm/bin/amdgpu-offload-arch"
        try:
            output = subprocess.check_output([command])
            return output.decode("utf-8").strip()
        except subprocess.CalledProcessError as e:
            error_message = f"Error: {e}"
            raise RuntimeError(error_message) from e
        except FileNotFoundError as e:
            # If the command is not found, print an error message
            error_message = f"The command {command} was not found."
            raise RuntimeError(error_message) from e

        return None

    def get_hipcc_rocm_version():
        # Run the hipcc --version command
        result = subprocess.run(
            ["hipcc", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Check if the command was executed successfully
        if result.returncode != 0:
            print("Error running 'hipcc --version'")
            return None

        # Extract the version using a regular expression
        match = re.search(r"HIP version: (\S+)", result.stdout)
        if match:
            # Return the version string
            return match.group(1)
        else:
            print("Could not find HIP version in the output")
            return None

    def get_nvcc_cuda_version(cuda_dir: str) -> Version:
        """Get the CUDA version from nvcc.

        Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
        """
        nvcc_output = subprocess.check_output(
            [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
        )
        output = nvcc_output.split()
        release_idx = output.index("release") + 1
        nvcc_cuda_version = parse(output[release_idx].split(",")[0])
        return nvcc_cuda_version

    def get_torch_arch_list() -> Set[str]:
        # TORCH_CUDA_ARCH_LIST can have one or more architectures,
        # e.g. "8.0" or "7.5,8.0,8.6+PTX". Here, the "8.6+PTX" option asks the
        # compiler to additionally include PTX code that can be runtime-compiled
        # and executed on the 8.6 or newer architectures. While the PTX code will
        # not give the best performance on the newer architectures, it provides
        # forward compatibility.
        env_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
        if env_arch_list is None:
            return set()

        # List are separated by ; or space.
        torch_arch_list = set(env_arch_list.replace(" ", ";").split(";"))
        if not torch_arch_list:
            return set()

        # Filter out the invalid architectures and print a warning.
        valid_archs = NVIDIA_SUPPORTED_ARCHS.union(
            {s + "+PTX" for s in NVIDIA_SUPPORTED_ARCHS}
        )
        arch_list = torch_arch_list.intersection(valid_archs)
        # If none of the specified architectures are valid, raise an error.
        if not arch_list:
            raise RuntimeError(
                "None of the CUDA architectures in `TORCH_CUDA_ARCH_LIST` env "
                f"variable ({env_arch_list}) is supported. "
                f"Supported CUDA architectures are: {valid_archs}."
            )
        invalid_arch_list = torch_arch_list - valid_archs
        if invalid_arch_list:
            warnings.warn(
                f"Unsupported CUDA architectures ({invalid_arch_list}) are "
                "excluded from the `TORCH_CUDA_ARCH_LIST` env variable "
                f"({env_arch_list}). Supported CUDA architectures are: "
                f"{valid_archs}.",
                stacklevel=2,
            )
        return arch_list

    # First, check the TORCH_CUDA_ARCH_LIST environment variable.
    compute_capabilities = get_torch_arch_list()
    if _is_cuda() and not compute_capabilities:
        # If TORCH_CUDA_ARCH_LIST is not defined or empty, target all available
        # GPUs on the current machine.
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            major, minor = torch.cuda.get_device_capability(i)
            if major < 7:
                raise RuntimeError(
                    "GPUs with compute capability below 7.0 are not supported."
                )
            compute_capabilities.add(f"{major}.{minor}")

    if _is_cuda():
        nvcc_cuda_version = get_nvcc_cuda_version(CUDA_HOME)
        if not compute_capabilities:
            # If no GPU is specified nor available, add all supported architectures
            # based on the NVCC CUDA version.
            compute_capabilities = NVIDIA_SUPPORTED_ARCHS.copy()
            if nvcc_cuda_version < Version("11.1"):
                compute_capabilities.remove("8.6")
            if nvcc_cuda_version < Version("11.8"):
                compute_capabilities.remove("8.9")
                compute_capabilities.remove("9.0")
        # Validate the NVCC CUDA version.
        if nvcc_cuda_version < Version("11.0"):
            raise RuntimeError("CUDA 11.0 or higher is required to build the package.")
        if nvcc_cuda_version < Version("11.1") and any(
            cc.startswith("8.6") for cc in compute_capabilities
        ):
            raise RuntimeError(
                "CUDA 11.1 or higher is required for compute capability 8.6."
            )
        if nvcc_cuda_version < Version("11.8"):
            if any(cc.startswith("8.9") for cc in compute_capabilities):
                # CUDA 11.8 is required to generate the code targeting compute capability 8.9.
                # However, GPUs with compute capability 8.9 can also run the code generated by
                # the previous versions of CUDA 11 and targeting compute capability 8.0.
                # Therefore, if CUDA 11.8 is not available, we target compute capability 8.0
                # instead of 8.9.
                warnings.warn(
                    "CUDA 11.8 or higher is required for compute capability 8.9. "
                    "Targeting compute capability 8.0 instead.",
                    stacklevel=2,
                )
                compute_capabilities = set(
                    cc for cc in compute_capabilities if not cc.startswith("8.9")
                )
                compute_capabilities.add("8.0+PTX")
            if any(cc.startswith("9.0") for cc in compute_capabilities):
                raise RuntimeError(
                    "CUDA 11.8 or higher is required for compute capability 9.0."
                )

        # Add target compute capabilities to NVCC flags.
        for capability in compute_capabilities:
            num = capability[0] + capability[2]
            NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=sm_{num}"]
            if capability.endswith("+PTX"):
                NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=compute_{num}"]

        # Use NVCC threads to parallelize the build.
        if nvcc_cuda_version >= Version("11.2"):
            nvcc_threads = int(os.getenv("NVCC_THREADS", 8))
            num_threads = min(os.cpu_count(), nvcc_threads)
            NVCC_FLAGS += ["--threads", str(num_threads)]

    elif _is_hip():
        amd_arch = get_amdgpu_offload_arch()
        if amd_arch not in ROCM_SUPPORTED_ARCHS:
            raise RuntimeError(
                f"Only the following arch is supported: {ROCM_SUPPORTED_ARCHS}"
                f"amdgpu_arch_found: {amd_arch}"
            )

    paged_attn_extension = CUDAExtension(
        name="fms.paged_c",
        sources=[
            "csrc/paged_attention/cache_kernels.cu",
            "csrc/paged_attention/attention/attention_kernels.cu",
            "csrc/paged_attention/cuda_utils_kernels.cu",
            "csrc/paged_attention/pybind.cpp",
        ],
        extra_compile_args={
            "cxx": CXX_FLAGS,
            "nvcc": NVCC_FLAGS,
        },
    )
    ext_modules.append(paged_attn_extension)
    cmdclass["build_ext"] = BuildExtension


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""
    with open(get_path("requirements.txt")) as f:
        requirements = f.read().strip().split("\n")
    return requirements


setup(
    name="ibm-fms",
    version="0.0.3",
    author="Brian Vaughan, Joshua Rosenkranz, Antoni Viros i Martin, Davis Wertheimer, Supriyo Chakraborty, Raghu Kiran Ganti",
    author_email="bvaughan@ibm.com, jmrosenk@us.ibm.com, aviros@ibm.com, Davis.Wertheimer@ibm.com, supriyo@us.ibm.com, rganti@us.ibm.com",
    description="IBM Foundation Model Stack",
    packages=find_packages(exclude=("csrc",)),
    install_requires=get_requirements(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    url="https://github.com/foundation-model-stack/foundation-model-stack",
    extras_require={"hf": ["transformers >= 4.35.0"]},
    license="Apache License 2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
)
