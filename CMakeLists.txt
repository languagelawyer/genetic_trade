cmake_minimum_required(VERSION 3.23)
project(rnn_trader LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast") # numpy is unhappy about -Ofast, but who cares, it is so good


add_library(rnn_trader SHARED)
target_sources(rnn_trader
	PRIVATE
		src/rnn_trader.cpp
		src/spot.cpp

	PUBLIC
	FILE_SET HEADERS
	BASE_DIRS src/
)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
	include(CTest)
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
	add_subdirectory(tests)
endif()

