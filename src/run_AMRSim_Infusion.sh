RUN_FILE="infusion/train_AMRInfusion.py"

MODEL_TYPE="infusion"
INFUSION_TYPE="co_attn_res"

YEAR="2024"
LANG="jp"
PARSER="spring"

LEARNING_RATE=0.001
NUM_EPS=10
DEVICE="cuda"


LOG_PREDS_ROOT="infusion/logs/emnlp/"
LOG_PREDS_FILE="$YEAR$LANG$PARSER$INFUSION_TYPE"
LOG_PREDS_PATH="$LOG_PREDS_ROOT$LOG_PREDS_FILE.txt"

LOG_MODELS_ROOT="infusion/ckpt/"
LOG_MODELS_FILE="$YEAR$LANG$PARSER$INFUSION_TYPE"
LOG_MODELS_PATH=$LOG_MODELS_ROOT$LOG_MODELS_FILE

mkdir -p $LOG_MODELS_PATH

PYTHONPATH=$WORKSPACE python $RUN_FILE \
    -model_type $MODEL_TYPE \
    -infusion_type $INFUSION_TYPE \
    -text_lang $LANG \
    -parser $PARSER \
    -data_year $YEAR \
    -learning_rate $LEARNING_RATE \
    -num_eps $NUM_EPS \
    -log_path $LOG_PREDS_PATH \
    -save_path $LOG_MODELS_PATH \
    -device $DEVICE \
    