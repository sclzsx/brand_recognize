# Brand recognization

## STEPS

### 1. Prepare videos.
Place videos named as '*_regular.mp4' or '*_fake.mp4' in a folder, where '*' indicates the brand name.

### 2. Generate dataset.
python prepare.py

### 3. Modify configurations.
Check cfg.py

### 4. Perform training.
python train.py

### 5. Perform evalution.
python eval.py

### 6. Run demo (make sure the camera is available first).
python demo.py