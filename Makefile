.PHONY: build-dependencies
build-dependencies:
	cd core && \
	cmake --fresh -S . -B build/debug \
	    -DBUILD_DEP_SHARE=OFF \
		-DBUILD_DEP_STATIC=ON && \
	cmake --build build/debug && \
	cmake --fresh -S . -B build/debug \
	    -DBUILD_DEP_SHARE=ON \
		-DBUILD_DEP_STATIC=OFF && \
	cmake --build build/debug

.PHONY: build-debug
build-debug:
	cd core && \
	cmake --fresh -S . -B build/debug --log-level=TRACE \
	    -DBUILD_SINGLE_PKG=OFF \
		-DBUILD_SHARED_LIBS=ON \
		-DCMAKE_BUILD_TYPE=Debug && \
	cmake --build build/debug && \
	ctest -VV --output-on-failure --test-dir build/debug
