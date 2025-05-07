set(Tests
	test_equity_history
)

foreach(Test ${Tests})
	add_executable(${Test} ${Test}.cpp)
	target_link_libraries(${Test} PRIVATE rnn_trader)
	add_test(${Test} ${Test})
endforeach()
