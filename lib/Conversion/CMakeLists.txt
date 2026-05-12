get_property(_dirs GLOBAL PROPERTY FLYDSL_BACKEND_LIB_CONVERSION_SUBDIRS)
foreach(_dir ${_dirs})
  add_subdirectory(${_dir})
endforeach()
