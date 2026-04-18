# APP

## Description
UI Application which includes model to classify sicks based on tear.

## MVP
First working application to show that we can do it and in which phase we are.

### CORE
- Add data from bmp +
- Preprocess data
- Run data through model
- Get Result
- Show feedback/graphs/data/probability

### Good to have
- Dedicated Backend server
- Save metrics/results and input image to the dedicated server
- Authentication

### Overkill
- Add 3d graph of crystal to application

## Output
### Data
imagename
datetime


### Classification
Healthy
Diabetes
Dry Eye Disease
Multiple Sclerosis
Primary Open-Angle Glaucoma

## How to run

**Requirements:** Python 3.11+

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file (see Config section below)

# 4. Run
python main.py
```

## Config

.env```
type="dev" // can be dev / prod / test
```