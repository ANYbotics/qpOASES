##
##     This file is part of qpOASES.
##
##     qpOASES -- An Implementation of the Online Active Set Strategy.
##     Copyright (C) 2007-2017 by Hans Joachim Ferreau, Andreas Potschka,
##     Christian Kirches et al. All rights reserved.
##
##     qpOASES is free software; you can redistribute it and/or
##     modify it under the terms of the GNU Lesser General Public
##     License as published by the Free Software Foundation; either
##     version 2.1 of the License, or (at your option) any later version.
##
##     qpOASES is distributed in the hope that it will be useful,
##     but WITHOUT ANY WARRANTY; without even the implied warranty of
##     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
##     See the GNU Lesser General Public License for more details.
##
##     You should have received a copy of the GNU Lesser General Public
##     License along with qpOASES; if not, write to the Free Software
##     Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
##

##
##     Filename:  CMakeLists.txt
##     Author:    Hans Joachim Ferreau (thanks to Milan Vukov), Aravind Vijayan (ANYbotics)
##     Version:   3.2.1
##

# NOTE(apoghosov): The comment above is just for historical reasons ... the package was migrated to ROS2 and the file
# was changed.

cmake_minimum_required(VERSION 3.16.3)
project(qp_oases)

set(CMAKE_CXX_STANDARD 17)

add_compile_options(-Wall -pedantic -Wfloat-equal -Wshadow -DLINUX -finline-functions -D__NO_COPYRIGHT__ -D__SUPPRESSANYOUTPUT__ -D__AVOID_LA_NAMING_CONFLICTS__)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if("$ENV{ROS_VERSION}" STREQUAL "1")

find_package(catkin REQUIRED)
catkin_package(
  INCLUDE_DIRS
    include
  LIBRARIES
    ${PROJECT_NAME}
  CATKIN_DEPENDS
)

add_library(${PROJECT_NAME}
  src/BLASReplacement.cpp
  src/Bounds.cpp
  src/Constraints.cpp
  src/Flipper.cpp
  src/Indexlist.cpp
  src/LAPACKReplacement.cpp
  src/Matrices.cpp
  src/MessageHandling.cpp
  src/Options.cpp
  src/OQPinterface.cpp
  src/QProblem.cpp
  src/QProblemB.cpp
  src/SolutionAnalysis.cpp
  src/SparseSolver.cpp
  src/SQProblem.cpp
  src/SQProblemSchur.cpp
  src/SubjectTo.cpp
  src/Utils.cpp
)

add_dependencies(${PROJECT_NAME}
  ${catkin_EXPORTED_TARGETS}
)

target_include_directories(${PROJECT_NAME} PRIVATE
  include
)

target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC
  ${catkin_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
)

install(
  TARGETS
  ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

############################################################
######################### examples #########################
############################################################

# compile qpOASES examples
SET(EXAMPLE_NAMES
  example1
  example1a
  example1b
  example2
  example3
  example3b
  example4
  example5
  exampleLP
  qrecipe
  qrecipeSchur
)

FOREACH (ELEMENT ${EXAMPLE_NAMES})
    add_executable(${PROJECT_NAME}_${ELEMENT}
      examples/${ELEMENT}.cpp
    )
    target_include_directories(${PROJECT_NAME}_${ELEMENT} PRIVATE
      include
    )
    target_include_directories(${PROJECT_NAME}_${ELEMENT} SYSTEM PUBLIC
      ${catkin_INCLUDE_DIRS}
    )
    target_link_libraries(${PROJECT_NAME}_${ELEMENT}
      ${PROJECT_NAME}
    )
    install(
      TARGETS
      ${PROJECT_NAME}_${ELEMENT}
      ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
      LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
      RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    )
ENDFOREACH (ELEMENT ${EXAMPLE_NAMES})


##########
## Test ##
##########
## GTest.
if(CATKIN_ENABLE_TESTING)
    catkin_add_gtest(test_${PROJECT_NAME}
      test/TestExample1.cpp
      test/TestExample1a.cpp
      test/TestExample1b.cpp
      test/TestExample2.cpp
      test/TestExample4.cpp
      test/TestExample5.cpp
      test/TestExample6.cpp
      test/TestExample7.cpp
      test/TestExampleLP.cpp
      test/TestExternalCholesky1.cpp
      test/TestGradientShift.cpp
      test/TestGuessedWS1.cpp
      test/TestIdentitySQProblem.cpp
      test/TestIndexList.cpp
      test/TestInfeasible1.cpp
      test/TestJanick1.cpp
      test/TestJanick2.cpp
      test/TestMatrices.cpp
      test/TestReleaseBug.cpp
      test/TestSmallSchur.cpp
      test/TestVanBareIsUnboundedQP.cpp
    )
    target_include_directories(test_${PROJECT_NAME} PRIVATE
      include
      test
    )
    target_include_directories(test_${PROJECT_NAME} SYSTEM PUBLIC
      ${catkin_INCLUDE_DIRS}
    )
    target_link_libraries(test_${PROJECT_NAME}
      gtest_main
      ${PROJECT_NAME}
    )

    # Generate test coverage report -> ${PROJECT_BINARY_DIR}/cmake_code_coverage/${PROJECT_NAME}/index.html
    find_package(cmake_code_coverage QUIET)
    if(cmake_code_coverage_FOUND)
        add_gtest_coverage(
          TEST_BUILD_TARGETS
          test_${PROJECT_NAME}
    )
    endif(cmake_code_coverage_FOUND)
endif()

###########
## Clang ##
###########

#find_package(cmake_clang_tools QUIET)
#if(cmake_clang_tools_FOUND)
#    add_default_clang_tooling()
#endif(cmake_clang_tools_FOUND)


else() # ROS version 2

set(PACKAGE_DEPENDENCIES
  )

find_package(ament_cmake REQUIRED)

foreach(PKG ${PACKAGE_DEPENDENCIES})
  find_package(${PKG} REQUIRED)
endforeach()

add_library(${PROJECT_NAME} SHARED
  src/BLASReplacement.cpp
  src/Bounds.cpp
  src/Constraints.cpp
  src/Flipper.cpp
  src/Indexlist.cpp
  src/LAPACKReplacement.cpp
  src/Matrices.cpp
  src/MessageHandling.cpp
  src/Options.cpp
  src/OQPinterface.cpp
  src/QProblem.cpp
  src/QProblemB.cpp
  src/SolutionAnalysis.cpp
  src/SparseSolver.cpp
  src/SQProblem.cpp
  src/SQProblemSchur.cpp
  src/SubjectTo.cpp
  src/Utils.cpp
  )

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
  )

ament_target_dependencies(${PROJECT_NAME} ${PACKAGE_DEPENDENCIES})

install(DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
  )

install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  )

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)

##########
## Test ##
##########

if(BUILD_TESTING)
  
  ############################################################
  ######################### examples #########################
  ############################################################

  # compile qpOASES examples
  # FIXME(apoghosov): do we really need to compile them?

  set(EXAMPLE_NAMES
    example1
    example1a
    example1b
    example2
    example3
    example3b
    example4
    example5
    exampleLP
    qrecipe
    qrecipeSchur
    )

  foreach(ELEMENT ${EXAMPLE_NAMES})

    add_executable(${PROJECT_NAME}_${ELEMENT}
      examples/${ELEMENT}.cpp
      )

    target_link_libraries(${PROJECT_NAME}_${ELEMENT}
      ${PROJECT_NAME}
      )

  endforeach(ELEMENT ${EXAMPLE_NAMES})

  find_package(ament_cmake_gtest REQUIRED)
  
  ament_add_gtest(test_${PROJECT_NAME}
    test/TestExample1.cpp
    test/TestExample1a.cpp
    test/TestExample1b.cpp
    test/TestExample2.cpp
    test/TestExample4.cpp
    test/TestExample5.cpp
    test/TestExample6.cpp
    test/TestExample7.cpp
    test/TestExampleLP.cpp
    test/TestExternalCholesky1.cpp
    test/TestGradientShift.cpp
    test/TestGuessedWS1.cpp
    test/TestIdentitySQProblem.cpp
    test/TestIndexList.cpp
    test/TestInfeasible1.cpp
    test/TestJanick1.cpp
    test/TestJanick2.cpp
    test/TestMatrices.cpp
    test/TestReleaseBug.cpp
    test/TestSmallSchur.cpp
    test/TestVanBareIsUnboundedQP.cpp
    )

  target_link_libraries(test_${PROJECT_NAME} ${PROJECT_NAME})

  # FIXME(apoghosov): is this the way we should do ROS2 code coverage
  find_package(cmake_code_coverage QUIET)

  if(cmake_code_coverage_FOUND)
    add_gtest_coverage(TEST_BUILD_TARGETS test_${PROJECT_NAME})
  endif(cmake_code_coverage_FOUND)

endif()

ament_package()

endif()
