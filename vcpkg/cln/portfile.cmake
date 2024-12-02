vcpkg_from_git(
    OUT_SOURCE_PATH SOURCE_PATH
    URL git://www.ginac.de/cln.git
    REF bc36e1e941c9296f37198c3125ac7f2b2ca4f48b  # cln_1-3-7
    PATCHES fix-build-path-independence.patch
)

vcpkg_cmake_configure(SOURCE_PATH "${SOURCE_PATH}")
vcpkg_cmake_install()

vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/${PORT})
vcpkg_fixup_pkgconfig()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/COPYING")
