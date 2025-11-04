import subprocess
import os
from termcolor import cprint

update = False
tests = [x.replace('.input.mlir', '') for x in sorted(os.listdir(".")) if x.endswith(".input.mlir")]

for test in tests:

    print(test, end='')
    with open(test + '.input.mlir', "r") as file:
        test_name = file.readlines()[0][3:].strip()
    # print(f"    {test_name}")

    env = {"TRITON_OVERRIDE_ARCH": "sm100"}
    subprocess.check_call(
        [
            "/workdir/git/build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt",
            "--tritongpu-partition-scheduling",
            "-allow-unregistered-dialect",
            test + ".input.mlir",
        ],
        env=env,
        stderr=subprocess.STDOUT,
    )
    actual_output = subprocess.check_output(
        [
            "/workdir/git/build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt",
            "--tritongpu-partition-scheduling",
            "-allow-unregistered-dialect",
            test + ".input.mlir",
        ],
        env=env,
        stderr=subprocess.STDOUT,
    ).decode('utf-8').strip()

    if update:
        with open(test + ".output.mlir", "w") as file:
            file.write(actual_output)

    with open(test + ".output.mlir", "r") as file:
        expected_output = file.read().strip()

    if actual_output == expected_output:
        cprint(' OK', 'green')
    else:
        cprint(' FAIL', 'red')
