cmake_minimum_required(VERSION 3.30)
project(Gern_Apps)


include_directories(common)

#Add the skeleton directory.
add_subdirectory(skeleton)
# All the apps.
add_subdirectory(blur)
add_subdirectory(softmax)