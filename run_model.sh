
model_type=wav2vec2-large-robust

model=AuxFormer

corpus=CREMA-D
num_classes=ALL #four or ALL
output_num=6
label_rule=M       #P, M, D
partition_number=1
data_mode=primary #primary or secondary
seed=0
label_type=categorical
label_learning=hard-label


corpus_type=${corpus}_${num_classes}_${data_mode}

# Training
python -u train.py \
--device            cuda \
--model_type        $model_type \
--lr                .725e-3 \
--corpus_type       $corpus_type \
--seed              $seed \
--epochs            30 \
--batch_size        32 \
--hidden_dim        1024 \
--num_layers        2 \
--output_num        $output_num \
--label_type        $label_type \
--label_learning    $label_learning \
--corpus            $corpus \
--num_classes       $num_classes \
--label_rule        $label_rule \
--partition_number  $partition_number \
--data_mode         $data_mode \
--model_path        model/${model_type}/${corpus_type}/${label_type}/${label_learning}/${data_mode}/${label_rule}/${model}/partition${partition_number}/seed_${seed}

## Evaluation
python -u test.py \
--device            cuda \
--model_type        $model_type \
--corpus_type       $corpus_type \
--seed              $seed \
--batch_size        1 \
--hidden_dim        1024 \
--num_layers        2 \
--output_num        $output_num \
--label_type        $label_type \
--label_learning    $label_learning \
--corpus            $corpus \
--num_classes       $num_classes \
--label_rule        $label_rule \
--partition_number  $partition_number \
--data_mode         $data_mode \
--model_path        model/${model_type}/${corpus_type}/${label_type}/${label_learning}/${data_mode}/${label_rule}/${model}/partition${partition_number}/seed_${seed}


