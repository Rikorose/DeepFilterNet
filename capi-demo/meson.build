project('demo', 'c')

cc = meson.get_compiler('c')
inc = include_directories('/usr/include/df/')
df = dependency('df', static: true)
executable('demo', 'main.c', dependencies: df, include_directories: [inc])
