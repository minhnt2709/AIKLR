
# A Co-Attentive Infusion of Structural Knowledge into Language Models towards Legal Textual Inference

## Models and Data
* Dataset: The Dataset is ONLY for research use and NOT for any commercial use. Please refer to [COLIEE](https://coliee.org/overview) for data. 
* Model: The checkpoint for the infusion model is available for download [here](https://1drv.ms/f/c/2b28b99fcdb225cc/Elw01jWTokxOgN5QlukkHpEBW21-dUBs03SwbkTjjPF49g?e=cxBipd).

## Usage
### AMR Parser
* Please refer to [SPRING](https://github.com/SapienzaNLP/spring) for implementing AMR parser.

### Infusion Model
- install requirements
```
pip install -r requirements.txt
```
Install [amr-utils](https://github.com/ablodge/amr-utils):
```
git clone https://github.com//ablodge/amr-utils
pip install penman
pip install ./amr-utils
```
- Train and Evaluate
```
bash run_AMRSim_Infusion.sh
```

## Citation

## Contact
In case of any concerns, please contact minhnt@jaist.ac.jp