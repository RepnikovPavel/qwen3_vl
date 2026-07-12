import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = ROOT / "docker" / "run.sh"


class DockerRunCachePolicyTest(unittest.TestCase):
    def test_tmpfs_cache_environment_uses_existing_home_as_pytorch_base(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models = root / "models"
            data = root / "data"
            binaries = root / "bin"
            arguments_path = root / "docker-arguments"
            models.mkdir()
            data.mkdir()
            binaries.mkdir()
            docker = binaries / "docker"
            docker.write_text(
                '#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$DOCKER_ARGS_FILE"\n',
                encoding="utf-8",
            )
            docker.chmod(0o755)
            environment = os.environ.copy()
            environment["PATH"] = f"{binaries}:{environment['PATH']}"
            environment["DOCKER_ARGS_FILE"] = str(arguments_path)

            completed = subprocess.run(
                [
                    str(RUN_SCRIPT),
                    "infer-cpu",
                    "--models",
                    str(models),
                    "--data",
                    str(data),
                    "--",
                    "--model",
                    "2b",
                    "--preflight-only",
                ],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            arguments = arguments_path.read_text(encoding="utf-8").splitlines()

        self.assertIn("/tmp:rw,nosuid,nodev,exec,size=8g,mode=1777", arguments)
        self.assertIn("HOME=/tmp", arguments)
        self.assertIn("TRITON_CACHE_DIR=/tmp/triton-cache", arguments)
        self.assertFalse(
            any(value.startswith("XDG_CACHE_HOME=") for value in arguments)
        )

    def test_demo_uses_persistent_state_without_data_mount(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models = root / "models"
            state = root / "state"
            binaries = root / "bin"
            arguments_path = root / "docker-arguments"
            models.mkdir()
            state.mkdir()
            binaries.mkdir()
            docker = binaries / "docker"
            docker.write_text(
                '#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$DOCKER_ARGS_FILE"\n',
                encoding="utf-8",
            )
            docker.chmod(0o755)
            environment = os.environ.copy()
            environment["PATH"] = f"{binaries}:{environment['PATH']}"
            environment["DOCKER_ARGS_FILE"] = str(arguments_path)
            environment["QWEN3_STATE"] = str(state)

            completed = subprocess.run(
                [
                    str(RUN_SCRIPT),
                    "demo",
                    "--models",
                    str(models),
                    "--port",
                    "8123",
                ],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            arguments = arguments_path.read_text(encoding="utf-8").splitlines()

        self.assertIn("127.0.0.1:8123:7860/tcp", arguments)
        self.assertIn(f"type=bind,src={models},dst=/models,readonly", arguments)
        self.assertIn(f"type=bind,src={state},dst=/state", arguments)
        self.assertIn("DEMO_STATE_DIR=/state", arguments)
        self.assertIn("python3", arguments)
        self.assertIn("demo.server", arguments)
        self.assertNotIn("/data", arguments)


if __name__ == "__main__":
    unittest.main()
