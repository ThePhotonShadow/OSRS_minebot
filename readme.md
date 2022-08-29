## Computer Vision OSRS Mining Bot

Originally a component of a data science final project.

## Dependencies

To use, download and extract the traned model data to the project directory such that the hierarchy is `root/newbuilt`.

https://drive.google.com/file/d/1gclgAFmKMdoofsyvI8YNnuS_EvtKqkH2/view?usp=sharing

## General Notes

- "script" is for data segmenting and can be ignored
- Execute contains the bot logic
- images contains the training and testing images
- LatexFinal contains the latex source
- mousedata contains the input data and KNN model
- newbuilt contains:
	- backup contains the trained weights at different intervals
	- CMakeFiles has what you expect
	- custom has my network definitions
	- images has the images again
	- labels has the labels again
	- the root directory has my compiled darknet binary and associated libraries and the initial pretrained weights