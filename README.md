# rubik-faces-extractor

A program for detecting colors and solving rubik cube

> **Notice:** If you want to detect only 8 colors around center -> *checkout to branch* **version-8-colors**

### Main folder: ras-detector:

#### ras-detector structure:

**detect_data** : Link to download below
|
output/* : Contains images of faces during detection
|
src/* : Contains source code
|
requirements.txt



### Starting here:

`~ git clone https://github.com/derektruong/rubik-faces-extractor.git`

`~ cd rubik-faces-extractor/ras-detector/`

`~ python -m venv venv`

`~ active pyenv (instrucion below)`

`~ pip install -r requirements.txt`

`~ python main.py`


### Running instruction:

- Show the face to be recognized and press the S key to recognize.
- If not recognized, re-align the position of the rubik's face and press the s key again.
- Show the Front, Back, Right, Left, Up, Down faces in turn as the label on the video cam and shoot until all 6 faces.
- Colors will be show up on figure and log in terminal.
- End.


> **Active python virtualenv:**

> -> Windows

> ~ source ./[venv_name]/Scripts/activate

> -> MacOS

> ~ source ./[venv_name]/bin/activate




### You can find <detect_data> folder here:

- **https://url.dscdut.com/dtc-face-data**
