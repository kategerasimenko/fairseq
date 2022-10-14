#!/bin/bash

while [[ $# -gt 0 ]]; do
key="$1"

case $key in
    -e|--expdir)
        EXPDIR="$2"
        shift
    ;;
    *)
        echo Unknown option '"'$key'"' >&2
        exit 1
        # unknown option
    ;;
esac
shift
done

REL_DIST=16
REL_OPT="--maximum-relative-position $REL_DIST"
RNN_OPT="--rnn-positional-embeddings"
NO_ABS_OPT="--no-token-positional-embeddings"

for encoding in "abs" "rel" "rnn"; do
    REL_OPT_SET=
    case encoding in
      rel) REL_OPT_SET=REL_OPT + " " + NO_ABS_OPT;;
      rnn) REL_OPT_SET=REL_OPT + " " + RNN_OPT;;
    esac

    for i in 30 60; do
        wrappers/train_transformer.sh \
            -e $EXPDIR \
            -r 42 \
            --emb-size 512 \
            --ffn-size 2048 \
            --att-heads 8 \
            --patience 10 \
            --depth 6 \
            --eval-dir custom_examples/translation/wmt20_encs.tgt \
            --tasks "czeng.$i" \
            --valid-tasks "newstest" \
            --shared-dict \
            --clip-norm 1.0 \
            --encoding $encoding \
            $REL_OPT_SET
    done

done
