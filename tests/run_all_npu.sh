#!/bin/bash
set -u
export ASCEND_RT_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1

TESTS="basic batch shm large object keys state_dict dws gloo npu_store keys_prefix rdma hccs ts_rdma"
PASS=0
FAIL=0

for t in $TESTS; do
    echo ""
    echo "=== GROUP: $t ==="
    timeout 120 python -u tests/test_torchstore_npu.py --only "$t" 2>&1 | grep -E '(\[PASS\]|\[FAIL\]|\[SKIP\]|Results:)'
    rc=${PIPESTATUS[0]}
    if [ $rc -eq 0 ]; then
        PASS=$((PASS+1))
    elif [ $rc -eq 124 ]; then
        echo "  [TIMEOUT] $t"
        FAIL=$((FAIL+1))
    else
        echo "  [EXIT $rc] $t"
        FAIL=$((FAIL+1))
    fi
    sleep 3
done

echo ""
echo "============================================"
echo "  FINAL: $PASS GROUP PASS / $FAIL GROUP FAIL (total $((PASS+FAIL)))"
echo "============================================"
