from pythonforandroid.recipe import Recipe
from pythonforandroid.toolchain import current_directory, shprint
import os

class PortaudioRecipe(Recipe):
    version = 'pa_stable_v190700_20210423'
    url = 'http://files.portaudio.com/archives/pa_stable_v190700_20210423.tgz'
    library = 'libportaudio.a'
    depends = []             # 순수 C 라이브러리
    conflicts = ['audio']
    # hostpython3 필요 없고, 타깃용만 빌드
    build_depends = []

    def get_recipe_env(self, arch):
        env = super().get_recipe_env(arch)
        # OpenSL ES 링크
        env['LDFLAGS'] = '-lOpenSLES'
        return env

    def build_arch(self, arch):
        build_dir = self.get_build_dir(arch.arch)
        with current_directory(build_dir):
            # configure + static-only
            shprint('./configure',
                    '--host={}'.format(arch.command_prefix),
                    '--disable-shared',
                    '--enable-static',
                    '--prefix={}'.format(build_dir),
                    # Android 에선 OpenSL ES 사용
                    'LDFLAGS=-lOpenSLES')
            shprint('make', '-j{}'.format(os.cpu_count()))
            shprint('make', 'install')

        # 설치된 include/lib 을 합쳐서 p4a 에 복사
        self.install_libs(arch)

recipe = PortaudioRecipe()
