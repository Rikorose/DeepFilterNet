project('demo', 'c')

cc = meson.get_compiler('c')
inc = include_directories('/usr/local/include/df/')
df = dependency('deepfilter', static: true)
executable('demo', 'main.c', dependencies: df, include_directories: [inc])
