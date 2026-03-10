## ex1

1. If you increase the width of the Lemniscate (increasing a), what issue can happen with the robot performing IK?
2. What can happen if you change the dt parameter in IK?
3. We implemented a simple numerical IK solver. What are the advantages and disadvantages compared to an analytical IK solver?
4. What are the limits of our IK solver compared to state-of-the-art IK solvers?



1. it might simply not reach the points on the edge of the lemniscate
2. `dt` controls the update of model position. it controls how far the joints can move within one iteration. When `dt` is too small, the model might never be able to reach the target destination. With larger `dt`, it might also not reach the target, but because it will not be precise enough. (this is ~ learning rate)
3. numerical solvers are usually faster, but might be prone to drifts and errors
4. it is really simple, quick to implement and it takes much less features into account, is probably much less efficient and less robust