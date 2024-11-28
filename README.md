<div align="center">

# A Distractor-Aware Memory for <br> Visual Object Tracking with SAM2

[Jovana Videnović](https://www.linkedin.com/in/jovana-videnovi%C4%87-5a5b08169/), [Alan Lukežič](https://www.vicos.si/people/alan_lukezic/), and [Matej Kristan](https://www.vicos.si/people/matej_kristan/)

Faculty of Computer and Information Science, University of Ljubljana

[[`Arxiv`](https://arxiv.org/abs/2411.17576)] [[`DiDi dataset`](#didi-a-distractor-distilled-dataset)] 



https://github.com/user-attachments/assets/d33dfea9-a27f-42a6-84f5-e69802372bda



https://github.com/user-attachments/assets/137566fa-09f8-40b2-aa20-349333d273dd






_The official implementation will be released soon._
</div>



## Abstract
Memory-based trackers such as SAM2 demonstrate remarkable performance, however still struggle with distractors. We propose a new plug-in distractor-aware memory (DAM) and management strategy that substantially improves tracking robustness. The new model is demonstrated on SAM2.1, leading to SAM2.1++, which sets a new state-of-the-art on six benchmarks, including the most challenging VOT/S benchmarks without additional training. We also propose a new distractor-distilled (DiDi) dataset to better study the distractor problem. See the [preprint](https://arxiv.org/abs/2411.17576) for more details.

## Getting Started
Code and detailed instructions for running it (including integration with [VOT toolkit](https://github.com/votchallenge/toolkit)) will be released soon.

## DiDi: A distractor-distilled dataset
DiDi is a distractor-distilled tracking dataset created to address the limitation of low distractor presence in current visual object tracking benchmarks. To enhance the evaluation and analysis of tracking performance amidst distractors, we have semi-automatically distilled several existing benchmarks into the DiDi dataset. The dataset is available for download at [this link](https://go.vicos.si/didi).

<p align="center"> <img src="imgs/didi-examples.jpg" width="80%"> </p>
<div align="center">
  <i>Example frames from the DiDi dataset showing challenging distractors. Targets are denoted by green bounding boxes.</i>
</div>

## Experimental results on DiDi

| Model         | Quality | Accuracy | Robustness |
|---------------|---------|----------|------------|
| TransT        | 0.465   | 0.669    | 0.678      |
| KeepTrack     | 0.502   | 0.646    | 0.748      |
| SeqTrack      | 0.529   | 0.714    | 0.718      |
| AQATrack      | 0.535   | 0.693    | 0.753      |
| AOT           | 0.541   | 0.622    | 0.852      |
| Cutie         | 0.575   | 0.704    | 0.776      |
| ODTrack       | 0.608   | 0.740 :1st_place_medal:	 | 0.809    |
| SAM2.1Long    | 0.646   | 0.719    | 0.883      |
| SAM2.1   | 0.649 :3rd_place_medal:	 | 0.720    | 0.887 :3rd_place_medal:	 |
| SAMURAI       | 0.680 :2nd_place_medal:	  | 0.722 :3rd_place_medal:	   | 0.930 :2nd_place_medal:	    |
| **SAM2.1++** (ours) | 0.694 :1st_place_medal:	 | 0.727 :2nd_place_medal:	 | 0.944 :1st_place_medal:	 |

## Acknowledgments

Our work is built on top of [SAM 2](https://github.com/facebookresearch/sam2?tab=readme-ov-file) by Meta FAIR.
