"""Guard against stale Debian package indexes in release build images."""
from pathlib import Path
import unittest


class DockerfileTests(unittest.TestCase):
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
