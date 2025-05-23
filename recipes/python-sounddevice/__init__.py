from pythonforandroid.recipe import PythonRecipe

class SoundDeviceRecipe(PythonRecipe):
    name = 'python-sounddevice'
    version = '0.4.4'
    url = 'https://github.com/spatialaudio/python-sounddevice/archive/v{version}.tar.gz'
    depends = ['portaudio', 'cffi', 'pycparser']
    call_hostpython_via_targetpython = False
    site_packages_name = 'sounddevice'

recipe = SoundDeviceRecipe()
