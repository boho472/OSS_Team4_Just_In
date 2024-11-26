<div align="center">

# A Distractor-Aware Memory for <br> Visual Object Tracking with SAM2

[Jovana Videnović](https://www.linkedin.com/in/jovana-videnovi%C4%87-5a5b08169/), [Alan Lukežič](https://www.vicos.si/people/alan_lukezic/), and [Matej Kristan](https://www.vicos.si/people/matej_kristan/)

Faculty of Computer and Information Science, University of Ljubljana

[[`Arxiv`]()] [[`DiDi dataset`](#didi-a-distractor-distilled-dataset)] 


https://github.com/user-attachments/assets/0bbeec33-2da9-4a90-85ae-52e92b0e1762



_The official implementation will be released soon._
</div>



## Abstract
Memory-based trackers use recent frames to model targets and localize them by comparing the current image with buffered frames. While achieving high performance, these trackers still struggle with distractors. We introduce a new distractor-aware memory model and an introspection-based update strategy for SAM2, called SAM2.1++, to enhance segmentation accuracy and tracking robustness. Additionally, we present the distractor-distilled DiDi dataset for better performance analysis in the presence of distractors. SAM2.1++ surpasses baseline SAM2.1 on seven benchmarks, setting new state-of-the-art results on six. 
For more details, please refer to the [preprint]().

## Getting Started
Code and detailed instructions for running it (including integration with [VOT toolkit](https://github.com/votchallenge/toolkit)) will be released soon.

## DiDi: A distractor-distilled dataset
DiDi is a distractor-distilled tracking dataset created to address the limitation of low distractor presence in current visual object tracking benchmarks. To enhance the evaluation and analysis of tracking performance amidst distractors, we have semi-automatically distilled several existing benchmarks into the DiDi dataset. The dataset will be available for download soon.

<p align="center"> <img src="imgs/didi-examples.jpg" width="80%"> </p>
<div align="center">
  <i>Example frames from the DiDi dataset showing challenging distractors. Targets are denoted by green bounding boxes.</i>
</div>

## Experimental results on DiDi

| Model         | Quality | Accuracy | Robustness |
|---------------|---------|----------|------------|
| SAMURAI       | 0.680 :2nd_place_medal:	  | 0.722 :3rd_place_medal:	   | 0.930 :2nd_place_medal:	    |
| SAM2.1Long    | 0.646   | 0.719    | 0.883      |
| ODTrack       | 0.608   | 0.740 :1st_place_medal:	 | 0.809    |
| Cutie         | 0.575   | 0.704    | 0.776      |
| AOT           | 0.541   | 0.622    | 0.852      |
| AQATrack      | 0.535   | 0.693    | 0.753      |
| SeqTrack      | 0.529   | 0.714    | 0.718      |
| KeepTrack     | 0.502   | 0.646    | 0.748      |
| TransT        | 0.465   | 0.669    | 0.678      |
| SAM2.1   | 0.649 :3rd_place_medal:	 | 0.720    | 0.887 :3rd_place_medal:	 |
| **SAM2.1++** (ours) | 0.694 :1st_place_medal:	 | 0.727 :2nd_place_medal:	 | 0.944 :1st_place_medal:	 |

## Acknowledgments

Our work is built on top of [SAM 2](https://github.com/facebookresearch/sam2?tab=readme-ov-file) by Meta FAIR.
