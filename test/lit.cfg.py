import os
import lit.formats

config.name = "FlyDSL"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".mlir"]

config.test_source_root = os.path.dirname(__file__)

project_dir = os.path.dirname(__file__) + "/.."
build_dir = os.environ.get(
    "FLY_BUILD_DIR", os.path.join(project_dir, "build-fly")
)

config.test_exec_root = os.path.join(build_dir, "test")
mlir_install = os.environ.get(
    "MLIR_PATH",
    os.path.join(project_dir, "..", "llvm-project", "build-flydsl", "mlir_install"),
)

config.substitutions.append(("%fly-opt", os.path.join(build_dir, "bin", "fly-opt")))
config.substitutions.append(
    ("%FileCheck", os.path.join(mlir_install, "bin", "FileCheck"))
)

config.environment["PATH"] = os.pathsep.join(
    [os.path.join(build_dir, "bin"), os.path.join(mlir_install, "bin")]
    + os.environ.get("PATH", "").split(os.pathsep)
)

if os.environ.get("SHOW_IR") or "show-ir" in lit_config.params:
    config.substitutions.insert(0, (r"\| FileCheck", "| tee /dev/stderr | FileCheck"))
