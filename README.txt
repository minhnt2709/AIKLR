This project uses code from open source project AMRSim (https://github.com/zzshou/AMRSim), AMRBART(https://github.com/goodbai-nlp/AMRBART).

1. To run the infusion architecture using AMRSim as the AMR encoding model
- install requirements from AMRSim https://github.com/zzshou/AMRSim
- cd AMRSim/sentence-transformers
- run sh run_train_SimInfusion.sh to train and validate models

2. To run the infusion architecture using AMRBART as the AMR encoding model
- install requirements from AMRBART https://github.com/goodbai-nlp/AMRBART
- cd AMRBART/pre_train
- To train and validate models
python infusion_AMRBART/train_AMRBARTInfusion.py -model_type=MODEL_TYPE -infusion_type=INFUSION_TYPE -text_lang=TEXT_LANG -parser=PARSER -data_year=YEAR -log_path=LOG_PATH -save_paht=MODEL_PATH