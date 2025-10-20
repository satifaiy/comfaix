# variable for models id

# paddle models usually need to convert to onnx
MODELS_PADDLE := PaddlePaddle/PP-OCRv5_mobile_det \
		  		 PaddlePaddle/en_PP-OCRv5_mobile_rec

MODELS_EFFICIENT_SAM := yunyangx/EfficientSAM

MODELS_VIT := onnx-community/dinov2-small

# use for conversion
PADDLE_2_ONNX := $(addsuffix .paddle,$(MODELS_PADDLE))

# all models that needed to download
MODELS := ${MODELS_PADDLE} ${MODELS_EFFICIENT_SAM} ${MODELS_VIT}

DOWNLOAD_MODELS := $(addsuffix .model,$(MODELS))

.PHONY: load-models
load-models: $(DOWNLOAD_MODELS) $(PADDLE_2_ONNX)
	@echo "models downloaded and converted"

%.model:
	[ -d "models/$*" ] || huggingface-cli download \
		--local-dir "models/$*" $*

%.paddle:
	paddle2onnx --model_dir "models/$*" \
            --model_filename inference.json \
            --params_filename inference.pdiparams \
            --save_file "models/$*/inference.onnx"

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
	TBB_ENABLE_SANITIZERS=1 ctest \
		-VV \
		--rerun-failed \
	    --output-on-failure --test-dir build/debug

.PHONY: build-release
build-release:
	cd core && \
	cmake --fresh -S . -B build/release --log-level=TRACE \
	    -DBUILD_SINGLE_PKG=ON \
		-DBUILD_SHARED_LIBS=ON \
		-DCMAKE_BUILD_TYPE=Release && \
	cmake --build build/release
