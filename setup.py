from setuptools import setup, Extension

go_board_ext = Extension(
    'go_board',
    sources=['go_board.c'],
    extra_compile_args=[
        '-O3',           # Maximum optimization
        '-march=native', # Use all CPU features available (AVX2, etc.)
        '-funroll-loops',
        '-ffast-math',
        '-Wall',
    ],
)

setup(
    name='go_board',
    version='1.0',
    description='High-performance Go board with Union-Find groups',
    ext_modules=[go_board_ext],
)
