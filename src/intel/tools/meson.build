# Copyright © 2017 Intel Corporation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

build_intel_tools = get_option('build-intel-tools')

aubinator = executable(
  'aubinator',
  files('aubinator.c', 'disasm.c', 'gen_batch_decoder.c',
        'gen_disasm.h', 'intel_aub.h'),
  dependencies : [dep_expat, dep_zlib, dep_dl, dep_thread, dep_m],
  include_directories : [inc_common, inc_intel],
  link_with : [libintel_common, libintel_compiler, libmesa_util],
  c_args : [c_vis_args, no_override_init_args],
  build_by_default : false,
)

aubinator_error_decode = executable(
  'aubinator_error_decode',
  files('aubinator_error_decode.c', 'disasm.c', 'gen_disasm.h',
        'gen_batch_decoder.c'),
  dependencies : [dep_zlib, dep_thread],
  include_directories : [inc_common, inc_intel],
  link_with : [libintel_common, libintel_compiler, libmesa_util],
  c_args : [c_vis_args, no_override_init_args],
  build_by_default : false,
)

i965_batchbuffer_logger = shared_library(
  'i965_batchbuffer_logger',
  files('i965_batchbuffer_logger.cpp', 'disasm.c', 'gen_disasm.h'),
  dependencies : [dep_zlib, dep_thread],
  include_directories : [inc_common, inc_intel],
  link_with : [libintel_common, libintel_compiler, libmesa_util, libnir, libisl],
  cpp_args : ['-std=c++11'],
  build_by_default : build_intel_tools,
  install : build_intel_tools,
)

i965_batchbuffer_dump_show = executable(
  'i965_batchbuffer_dump_show',
  files('i965_batchbuffer_dump_show.c'),
  include_directories : [inc_common, inc_intel],
  c_args : [],
  build_by_default : build_intel_tools,
  install : build_intel_tools,
)

i965_batchbuffer_dump_show_xml = executable(
  'i965_batchbuffer_dump_show_xml',
  files('i965_batchbuffer_dump_show_xml.cpp'),
  include_directories : [inc_common, inc_intel],
  cpp_args : [],
  build_by_default : build_intel_tools,
  install : build_intel_tools,
)

i965_batchbuffer_dump_show_json = executable(
  'i965_batchbuffer_dump_show_json',
  files('i965_batchbuffer_dump_show_json.cpp'),
  include_directories : [inc_common, inc_intel],
  cpp_args : [],
  build_by_default : build_intel_tools,
  install : build_intel_tools,
)

gen_shader_disassembler = executable(
  'gen_shader_disassembler',
  files('gen_shader_disassembler.c', 'disasm.c'),
  dependencies : [dep_thread],
  include_directories : [inc_common, inc_intel],
  link_with : [libintel_common, libintel_compiler, libmesa_util],
  cpp_args : [],
  build_by_default : build_intel_tools,
  install : build_intel_tools,
)

libdir_value = join_paths(get_option('prefix'), get_option('libdir'))
sed_replace_string = 's,[@]libdir[@],' + libdir_value + ',g'
i965_batchbuffer_logger_sh = custom_target(
  'i965_batchbuffer_logger_sh',
  input : 'i965_batchbuffer_logger_sh.in',
  output : 'i965_batchbuffer_logger_sh',
  command : ['sed', '-e', sed_replace_string, '@INPUT@'],
  capture : true,
  build_by_default : build_intel_tools,
  install : build_intel_tools,
  install_dir : get_option('bindir'),
)

if build_intel_tools
  i965_batchbuffer_logger_headers = install_headers(
    'i965_batchbuffer_logger_app.h',
    'i965_batchbuffer_logger_output.h'
  )
endif
