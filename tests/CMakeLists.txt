add_executable(test_sinc test_sinc.cpp)
target_link_libraries(test_sinc PRIVATE sar_focuser)
add_test(NAME test_sinc COMMAND test_sinc)

add_executable(test_fft test_fft.cpp)
target_link_libraries(test_fft PRIVATE sar_focuser)
add_test(NAME test_fft COMMAND test_fft)

add_executable(test_types test_types.cpp)
target_link_libraries(test_types PRIVATE sar_focuser)
add_test(NAME test_types COMMAND test_types)

add_executable(test_pipeline test_pipeline.cpp)
target_link_libraries(test_pipeline PRIVATE sar_focuser)
add_test(NAME test_pipeline COMMAND test_pipeline)

add_executable(test_polsar test_polsar.cpp)
target_link_libraries(test_polsar PRIVATE sar_focuser)
add_test(NAME test_polsar COMMAND test_polsar)
