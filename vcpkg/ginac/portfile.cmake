vcpkg_from_git(
    OUT_SOURCE_PATH SOURCE_PATH
    URL git://www.ginac.de/ginac.git
    REF 29bdf53d3b8b71410e2fdee87dd8d8eecf439ece  # release_1-8-7
    PATCHES fix-linking.patch
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DCMAKE_DISABLE_FIND_PACKAGE_BISON=TRUE
        -DCMAKE_DISABLE_FIND_PACKAGE_FLEX=TRUE
        -DCMAKE_DISABLE_FIND_PACKAGE_Readline=TRUE
)
vcpkg_cmake_install()

vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/${PORT})
vcpkg_fixup_pkgconfig()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/COPYING")
