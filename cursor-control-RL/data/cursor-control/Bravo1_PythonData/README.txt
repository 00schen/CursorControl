Converted BCI data:

Directory structure:
  /ConvertedData/{ExperiemntTime}/{ExperiemntNumber}.pkl

Each file {ExperiemntNumber}.pkl is a pickled file with the following variables:
  - TargetAngle: angular position from the center in degrees
  - TargetPosition: target position on the screen
  - CursorState: (5 x time array) with info about actual cursor state. Rows contain:
      1) X position
      2) Y position
      3) X Velocity
      4) Y velocity
      5) baseline offset
        Further details found here: https://www.nature.com/articles/nn.3265
  - IntendedCursorState: same structure as CursorState, but without assist
  - CursorAssist: (time, ) with assist level at each time. 1 = full asssist, 0 = no assist
  - NeuralFeatures: (n_features, time) ndarray of filtered ECoG data
  - Events: (2, ) ndarray with trial event timing info
    - Events[0] = Inter Trial Interval end time (i.e. time user begins controlling the cursor)
    - Events[1] = Time target is reached
  - NeuralTime: Time  corresponding to the end of each bin of data in NeuralFeatures. So the index of the start of the trial is np.argmin(np.abs(NeuralTime - InterTrialInterval))
