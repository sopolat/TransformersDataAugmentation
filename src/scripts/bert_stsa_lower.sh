#!/usr/bin/env bash

SRC=/content/TransformersDataAugmentation/src
CACHE=/content/CACHE
TASK=stsa
BERTLR=0.00004
for NUMEXAMPLES in 10;
do
    for i in {0..0};
        do
        RAWDATADIR=/content/datasets/${TASK}/exp_${i}_${NUMEXAMPLES}

       # Baseline classifier
        python $SRC/bert_aug/bert_classifier.py --task $TASK  --data_dir $RAWDATADIR --seed ${i} --learning_rate $BERTLR --cache $CACHE > $RAWDATADIR/bert_baseline.log

      ##############
      ## EDA
      ##############

      EDADIR=$RAWDATADIR/eda
      mkdir $EDADIR
      python $SRC/bert_aug/eda.py --input $RAWDATADIR/train.tsv --output $EDADIR/eda_aug.tsv --num_aug=1 --alpha=0.1 --seed ${i}
      cat $RAWDATADIR/train.tsv $EDADIR/eda_aug.tsv > $EDADIR/train.tsv
      cp $RAWDATADIR/test.tsv $EDADIR/test.tsv
      cp $RAWDATADIR/dev.tsv $EDADIR/dev.tsv
      python $SRC/bert_aug/bert_classifier.py --task $TASK --data_dir $EDADIR --seed ${i} --learning_rate $BERTLR --cache $CACHE  > $RAWDATADIR/bert_eda.log


        #######################
        # GPT2 Classifier
        #######################

        GPT2DIR=$RAWDATADIR/gpt2
        mkdir $GPT2DIR
        python $SRC/bert_aug/cgpt2.py --data_dir $RAWDATADIR --output_dir $GPT2DIR --task_name $TASK  --num_train_epochs 25 --seed ${i} --top_p 0.9 --temp 1.0 --cache $CACHE
        cat $RAWDATADIR/train.tsv $GPT2DIR/cmodgpt2_aug_3.tsv > $GPT2DIR/train.tsv
        cp $RAWDATADIR/test.tsv $GPT2DIR/test.tsv
        cp $RAWDATADIR/dev.tsv $GPT2DIR/dev.tsv
        python $SRC/bert_aug/bert_classifier.py --task $TASK --data_dir $GPT2DIR --seed ${i} --learning_rate $BERTLR  --cache $CACHE > $RAWDATADIR/bert_gpt2_3.log

    #    #######################
    #    # Backtranslation DA Classifier
    #    #######################

    BTDIR=$RAWDATADIR/bt
    mkdir $BTDIR
    python $SRC/bert_aug/backtranslation.py --data_dir $RAWDATADIR --output_dir $BTDIR --task_name $TASK  --seed ${i} --cache $CACHE
    cat $RAWDATADIR/train.tsv $BTDIR/bt_aug.tsv > $BTDIR/train.tsv
    cp $RAWDATADIR/test.tsv $BTDIR/test.tsv
    cp $RAWDATADIR/dev.tsv $BTDIR/dev.tsv
    python $SRC/bert_aug/bert_classifier.py --task $TASK --data_dir $BTDIR --seed ${i} --learning_rate $BERTLR  --cache $CACHE  > $RAWDATADIR/bert_bt.log

   # #######################
   # # CBERT Classifier
   # #######################

    CBERTDIR=$RAWDATADIR/cbert
    mkdir $CBERTDIR
    python $SRC/bert_aug/cbert.py --data_dir $RAWDATADIR --output_dir $CBERTDIR --task_name $TASK  --num_train_epochs 10 --seed ${i}  --cache $CACHE > $RAWDATADIR/cbert.log
    cat $RAWDATADIR/train.tsv $CBERTDIR/cbert_aug.tsv > $CBERTDIR/train.tsv
    cp $RAWDATADIR/test.tsv $CBERTDIR/test.tsv
    cp $RAWDATADIR/dev.tsv $CBERTDIR/dev.tsv
    python $SRC/bert_aug/bert_classifier.py --task $TASK --data_dir $CBERTDIR --seed ${i} --learning_rate $BERTLR  --cache $CACHE > $RAWDATADIR/bert_cbert.log

    done
done


