"""Guard against stale Debian package indexes in release build images."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest


class DockerfileTests(unittest.TestCase):
    def test_linux_images_install_the_release_rust_toolchain(self):
        root = Path(__file__).resolve().parents[1]
        for name in ('Dockerfile_x86', 'Dockerfile_arm'):
            with self.subTest(dockerfile=name):
                lines = (root / 'docker' / name).read_text(encoding='utf-8').splitlines()
                install = next(line for line in lines if line.startswith('RUN curl ') and 'sh.rustup.rs' in line)
                self.assertIn('--default-toolchain 1.97.1', install)

    def test_linux_build_scripts_enforce_the_lockfile(self):
        bash = shutil.which('bash')
        if os.name == 'nt':
            git_bash = Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Git/bin/bash.exe'
            if git_bash.exists():
                bash = str(git_bash)
        if not bash:
            self.skipTest('bash is required to exercise Linux build scripts')
        root = Path(__file__).resolve().parents[1]
        for name in ('docker-build-x86.sh', 'docker-build-arm.sh'):
            with self.subTest(script=name), tempfile.TemporaryDirectory() as directory:
                temp = Path(directory)
                fake_bin = temp / 'bin'
                fake_bin.mkdir()
                cargo = fake_bin / 'cargo'
                cargo.write_text('#!/bin/sh\nprintf "CARGO_ARGS:%s\\n" "$*" >&2\nexit 73\n', encoding='utf-8')
                cargo.chmod(0o755)
                path = str(temp).replace('\\', '/')
                if os.name == 'nt':
                    path = '/' + path[0].lower() + path[2:]
                script = (root / 'docker' / name).read_text(encoding='utf-8')
                script = script.replace('/src', path + '/src').replace('/build', path + '/build')
                env = dict(os.environ, PATH=str(fake_bin) + os.pathsep + os.environ['PATH'])
                result = subprocess.run([bash, '-c', script], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 73, result.stderr)
                self.assertIn('CARGO_ARGS:build --locked --release --target ', result.stderr)

    def test_package_indexes_are_refreshed_in_the_install_layer(self):
        for name in ('Dockerfile_x86', 'Dockerfile_arm'):
            with self.subTest(name=name):
                lines = (Path(__file__).resolve().parents[1] / 'docker' / name).read_text().splitlines()
                layers = []
                for line in lines:
                    if line.startswith('RUN '):
                        layers.append(line)
                    elif layers and line.startswith(' '):
                        layers[-1] += line
                install = next(layer for layer in layers if 'apt-get' in layer and ' install ' in layer)
                self.assertIn(' update ', install)
                self.assertLess(install.index(' update '), install.index(' install '))


if __name__ == '__main__':
    unittest.main()
