TO DO List:
1. Run a more robust experiment
2. Try running bayesian lora with bayesian_lora library properly, optimize GPU memory usage (failing because of vocab. size).
3. Add pipeline for bayesian_lora similar to the one for laplace.
4. Estimate EU as maximal and the minimal Shannon entropy within the credal set, not just the mean entropy.
5. Compare the results with other uncertainty estimation methods (e.g., MC Dropout, Deep Ensembles).
6. Test on a wider range of adversarial prompts and datasets.