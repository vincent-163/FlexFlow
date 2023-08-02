#! /usr/bin/env bash
set -x
set -e

cleanup() {
    rm -rf ../../inference/prompt ../../inference/weights ../../inference/tokenizer ../../inference/output
}

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

# Run C++ tests instead of Python tests, if desired
CPP_INFERENCE_TESTS=${CPP_INFERENCE_TESTS:-OFF}

# Run C++ tests instead of 
if [ "$CPP_INFERENCE_TESTS" = "ON" ]; then
    ./inference/cpp_inference_tests.sh
    exit 0
fi

# Clean up before test (just in case)
cleanup

# Update the transformers library to support the LLAMA model
pip3 install --upgrade transformers sentencepiece

# Create test prompt file
mkdir -p ../../inference/prompt
echo '["Give three tips for staying healthy."]' > ../../inference/prompt/test.json

# Create output folder
mkdir -p ../../inference/output

###############################################################################################
############################ Speculative inference tests ######################################
###############################################################################################

python ../inference/python/spec_infer.py -config-file ./inference/test_configs/spec-infer_llama_full-precision_tp1_pp4.json
python ../inference/python/spec_infer.py -config-file ./inference/test_configs/spec-infer_llama_full-precision_tp2_pp2.json
python ../inference/python/spec_infer.py -config-file ./inference/test_configs/spec-infer_llama_half-precision_tp1_pp4.json
python ../inference/python/spec_infer.py -config-file ./inference/test_configs/spec-infer_llama_half-precision_tp2_pp2.json

python ../inference/python/spec_infer.py -config-file ./inference/test_configs/spec-infer_opt_full-precision_tp1_pp4.json
python ../inference/python/spec_infer.py -config-file ./inference/test_configs/spec-infer_opt_full-precision_tp2_pp2.json
python ../inference/python/spec_infer.py -config-file ./inference/test_configs/spec-infer_opt_half-precision_tp1_pp4.json
python ../inference/python/spec_infer.py -config-file ./inference/test_configs/spec-infer_opt_half-precision_tp2_pp2.json

###############################################################################################
############################ Incremental decoding tests #######################################
###############################################################################################

# LLAMA, full precision
python ../inference/python/incr_decoding.py -config-file ./inference/test_configs/incr-decoding_llama-small_full-precision_tp1_pp4.json
python ../inference/python/incr_decoding.py -config-file ./inference/test_configs/incr-decoding_llama-large_full-precision_tp1_pp4.json
python ../inference/python/incr_decoding.py -config-file ./inference/test_configs/incr-decoding_llama-small_full-precision_tp2_pp2.json
python ../inference/python/incr_decoding.py -config-file ./inference/test_configs/incr-decoding_llama-large_full-precision_tp2_pp2.json
python ../inference/python/incr_decoding.py -config-file ./inference/test_configs/incr-decoding_llama-small_full-precision_tp4_pp1.json
python ../inference/python/incr_decoding.py -config-file ./inference/test_configs/incr-decoding_llama-large_full-precision_tp4_pp1.json
# LLAMA, half precision
python ../inference/python/incr_decoding.py -config-file ./inference/test_configs/incr-decoding_llama-small_half-precision_tp1_pp4.json
python ../inference/python/incr_decoding.py -config-file ./inference/test_configs/incr-decoding_llama-large_half-precision_tp1_pp4.json
python ../inference/python/incr_decoding.py -config-file ./inference/test_configs/incr-decoding_llama-small_half-precision_tp2_pp2.json
python ../inference/python/incr_decoding.py -config-file ./inference/test_configs/incr-decoding_llama-large_half-precision_tp2_pp2.json
python ../inference/python/incr_decoding.py -config-file ./inference/test_configs/incr-decoding_llama-small_half-precision_tp4_pp1.json
python ../inference/python/incr_decoding.py -config-file ./inference/test_configs/incr-decoding_llama-large_half-precision_tp4_pp1.json
# OPT, full precision
python ../inference/python/incr_decoding.py -config-file ./inference/test_configs/incr-decoding_opt-small_full-precision_tp1_pp4.json
python ../inference/python/incr_decoding.py -config-file ./inference/test_configs/incr-decoding_opt-large_full-precision_tp1_pp4.json
python ../inference/python/incr_decoding.py -config-file ./inference/test_configs/incr-decoding_opt-small_full-precision_tp2_pp2.json
python ../inference/python/incr_decoding.py -config-file ./inference/test_configs/incr-decoding_opt-large_full-precision_tp2_pp2.json
python ../inference/python/incr_decoding.py -config-file ./inference/test_configs/incr-decoding_opt-small_full-precision_tp4_pp1.json
python ../inference/python/incr_decoding.py -config-file ./inference/test_configs/incr-decoding_opt-large_full-precision_tp4_pp1.json
# OPT, half precision
python ../inference/python/incr_decoding.py -config-file ./inference/test_configs/incr-decoding_opt-small_half-precision_tp1_pp4.json
python ../inference/python/incr_decoding.py -config-file ./inference/test_configs/incr-decoding_opt-large_half-precision_tp1_pp4.json
python ../inference/python/incr_decoding.py -config-file ./inference/test_configs/incr-decoding_opt-small_half-precision_tp2_pp2.json
python ../inference/python/incr_decoding.py -config-file ./inference/test_configs/incr-decoding_opt-large_half-precision_tp2_pp2.json
python ../inference/python/incr_decoding.py -config-file ./inference/test_configs/incr-decoding_opt-small_half-precision_tp4_pp1.json
python ../inference/python/incr_decoding.py -config-file ./inference/test_configs/incr-decoding_opt-large_half-precision_tp4_pp1.json
