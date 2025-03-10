import pandas as pd
import random as rand
import numpy as np

itteration = 100

countData = pd.DataFrame(0, index = np.arange(itteration), columns = ['Herald', 'Detector1', 'Detector2', 'CorrectionH', 'Correction1', 'Correction2'])

def interactionChain(increment):
    heraldRoll = rand.randint(0, 100)
    results = [0,0,0,0,0,0]
    # The chance of the Herald clicking when a photon is absorbed
    if heraldRoll < 98:
        results[0] = 1
    # else a click is missed
    else:
        results[3] = 1

    # The chance of the Herald click being a false positive, can only occur when we have a herald
    if results[0]:
        heraldRoll = rand.randint(0, 100)
        if heraldRoll > 98:
            results[3] = 1
            countData.loc[increment] = results
            return
    # The cases that can occur
    # [0, 1] no Herald, it was a missed photon
    # [1, 1] Herald, it was a false positive
    # [1, 0] Herald, is actually a click



    # For the classical regime Detector 1 and 2 have a probability of clicking at 50% each
    # They will be 48 < x < 52 because the margin of 2% either side accounts for a missed photon
    detectorRoll = rand.randint(0, 100)
    if detectorRoll <= 48:
        results[1] = 1
    # if below 50 but greater than 48, the tick was missed
    elif detectorRoll <= 50 and detectorRoll >48:
        results[4] = 1

    elif detectorRoll >= 52:
        results[2] = 1
    # if below 52 but greater than 50, the tick was missed
    elif detectorRoll <= 52 and detectorRoll >50:
        results[5] = 1


    # for both detectors we now need a probability of ticking falsely
    # This is completely irrespective of any other result so far
    # Correction 1
    detectorRoll = rand.randint(0, 100)
    # 5 is comparatively big, for proof of concept
    if detectorRoll < 5:
        results[1] = 1
        results[4] = 1
    # Correction 2
    detectorRoll = rand.randint(0, 100)
    if detectorRoll < 5:
        results[2] = 1
        results[5] = 1

    # Possible cases
    # [0,0,1,0] no detections because detector 1 missed it
    # [0,0,0,1] no detections because detector 2 missed it
    # [1,0,1,0] detection for chanel 1, but it is a false click
    # [0,1,0,1] detection for chanel 2, but it is a false click
    # [1,1,1,1] detection for both but, both are false clicks
    # [1,0,0,0] detection for chanel 1, is correct
    # [0,1,0,0] detection for chanel 2, is correct
    countData.loc[increment] = results

for x in range(itteration):
    interactionChain(x)

print(countData)