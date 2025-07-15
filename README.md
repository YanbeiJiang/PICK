# PICK
Repository for "Reasoning Like Experts: Leveraging Multimodal Large Language Models for Drawing-based Psychoanalysis"

## Abstract
Multimodal Large Language Models (MLLMs) have demonstrated exceptional performance across various objective multimodal perception tasks, yet their application to subjective, emotionally nuanced domains, such as psychological analysis, remains largely unexplored. In this paper, we introduce PICK, a multi-step framework designed for Psychoanalytical Image Comprehension through hierarchical analysis and Knowledge injection with MLLMs, specifically focusing on the House-Tree-Person (HTP) Test, a widely used psychological assessment in clinical practice. First, we decompose drawings containing multiple instances into semantically meaningful sub-drawings, constructing a hierarchical representation that captures spatial structure and content across three levels: single-object level, multi-object level, and whole level. Next, we analyze these sub-drawings at each level with a targeted focus, extracting psychological or emotional insights from their visual cues. We also introduce an HTP knowledge base and design a feature extraction module, trained with reinforcement learning, to generate a psychological profile for single-object level analysis. This profile captures both holistic stylistic features and dynamic object-specific features (such as those of the house, tree, or person), correlating them with psychological states. Finally, we integrate these multi-faceted information to produce a well-informed assessment that aligns with expert-level reasoning. Our approach bridges the gap between MLLMs and specialized expert domains, offering a structured and interpretable framework for understanding human mental states through visual expression. Experimental results demonstrate that the proposed PICK significantly enhances the capability of MLLMs in psychological analysis. It is further validated as a general framework through extensions to emotion understanding tasks.

<img src="figures/intro_new-1.png" width="600"> 
<img src="figures/task_definition-1.png" width="600"> 


## Dataset
### Download
**You may direct download MultiStAR from [here](https://drive.google.com/file/d/1TQLD4pK7C7ERM5qMa7YCey0Sq9_MMs9J/view?usp=drive_link).**

**Alternatively, you may run the code to generate the dataset by following the [instructions](#Setup).**

### Dataset Structure Description
#### Direct Answer (under "dataset" folder): 

This dataset contains seven Configurations (same as RAVEN) and is organized in JSON format and contains the following main fields:

- **`question_num`**: The total number of questions in the current sample.
- **`rules`**: Defines the logical rules applied to the visual patterns (e.g., shape progression, color consistency).
- **`panels`**: Describes each visual panel, including attributes like position, shape, size, and color of the objects.

Each **`questions`** entry contains the following fields:

- **`filename`**: The name of the source annotation file (e.g., from RAVEN or other templates).
- **`question`**: The question.
- **`answer`**: The correct answer choice label (e.g., "A", "B", "C", or "D").
- **`template_filename`**: The filename of the template used to generate the question.
- **`choices`**: A list of multiple-choice answer options, each prefixed with a label (e.g., "A: 7").
- **`config`**: The configuration of the panel layout, such as object position or type (e.g., "center_single").
- **`image_filename`**: The path to the panel image associated with the question.

#### Logical Chain (in the file "logical_chain_questions.json")

- **`question`**: The question.
- **`choices`**: A list of answer options, each labeled (e.g., `"A: 1"`).
- **`correct_answer`**: The correct answer label (e.g., `"A"`, `"B"`).
- **`config`**: RAVEN configurations.
- **`image_path`**: Image path.
- **`stage`**: Indicates the subtask or reasoning stage (e.g., `"single_panel_1_left"`), useful for multi-stage reasoning pipelines.
- **`attribute`**: The target visual attribute in the question (e.g., `"number"`, `"position"`, `"shape"`).

## Setup
### Requirements
* cv2
* numpy
* tqdm
### File and Package Required
Download original RAVEN dataset and unzip it under the **`./RAVEN`** directory [RAVEN.zip](https://drive.google.com/file/d/1rmg_Eavn-EZ5bas4XI4yIWFIV-3fJ-M4/view?usp=sharing).

### Run
Run **`./generate_dataset.sh`**

## 📚 Citation

If you find this dataset useful in your research or work, please consider citing:

```bibtex
@misc{jiang2025perceptionevaluatingabstractvisual,
      title={Beyond Perception: Evaluating Abstract Visual Reasoning through Multi-Stage Task}, 
      author={Yanbei Jiang and Yihao Ding and Chao Lei and Jiayang Ao and Jey Han Lau and Krista A. Ehinger},
      year={2025},
      eprint={2505.21850},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.21850}, 
}



